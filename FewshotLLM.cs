using Azure.AI.OpenAI;
using InferenceTuning.Contract;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using Serilog;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace InferenceTuning
{
    public class AutoLLM
    {
        private Func<MLContext, IDataView, double> _evaluatorFunc;
        private Func<MLContext, IEnumerable<object>, Parameter, IEstimator<ITransformer>> _pipelineFunc;
        private IEnumerable<object> _examples;
        private MLContext _context;
        private Random _random = new Random();
        private SearchSpace _searchSpace;

        public AutoLLM(MLContext context)
        {
            _context = context;
        }

        public AutoLLM SetEvaluator(Func<MLContext, IDataView, double> evaluator)
        {
            _evaluatorFunc = evaluator;

            return this;
        }

        public AutoLLM SetPipeline<TExample, TOption>(Func<MLContext, IEnumerable<TExample>, TOption, IEstimator<ITransformer>> func, IEnumerable<TExample> examples, SearchSpace<TOption> searchSpace)
            where TOption : class, new()
        {
            var option = searchSpace.SampleFromFeatureSpace(searchSpace.Default);
            _pipelineFunc = (context, objs, option) => func(context, objs.Cast<TExample>(), option.AsType<TOption>());
            _examples = examples.Select(x => (object)x!);
            _searchSpace = searchSpace;

            return this;
        }

        public ITransformer Fit(IDataView trainSet, IDataView validationSet)
        {
            // stage 1. select examples.
            // for now, just randomly select N examples.
            var selectedExamples = new List<object>();
            var exampleCount = _examples.Count();
            var defaultParameter = _searchSpace.SampleFromFeatureSpace(_searchSpace.Default);
            var bestScore = double.MinValue;
            for (int i = 0; i <= Math.Sqrt(exampleCount); ++i)
            {
                var examplesToPick = _random.Next(1, exampleCount);
                var pickedExamples = _examples.OrderBy(x => _random.Next()).Take(examplesToPick).ToArray();
                Log.Debug($"Select examples: {JsonSerializer.Serialize(pickedExamples)}");
                var pipeline = _pipelineFunc(_context, pickedExamples, defaultParameter);
                var model = pipeline.Fit(trainSet);
                var eval = model.Transform(validationSet);
                var score = _evaluatorFunc(_context, eval);
                if (score > bestScore)
                {
                    bestScore = score;
                    selectedExamples = pickedExamples.ToList();
                }
            }

            Log.Information($"Best score after example selection: {bestScore}");
            Log.Information($"Best examples: {JsonSerializer.Serialize(selectedExamples)}");


            // stage 2. Inference parameter tuning.
            var autoML = _context.Auto().CreateExperiment();
            var trialRunner = new AutoLLMInferenceParameterTrialRunner(
                _context,
                trainSet,
                validationSet,
                _evaluatorFunc,
                (context, parameter) => _pipelineFunc(context, selectedExamples, parameter));
            autoML.SetTrialRunner(trialRunner)
                  .SetCostFrugalTuner()
                  .AddSearchSpace("llm", _searchSpace)
                  .SetMaxModelToExplore(10);

            var result = autoML.Run();
            Log.Information($"Best score after inference parameter tuning: {result.Metric}");
            return result.Model;
        }

        public async static Task Example1()
        {
            var context = new MLContext();
            string level = "Level 4";
            // setup log
            Log.Logger = new LoggerConfiguration()
                            .MinimumLevel.Debug()
                            .WriteTo.Console()
                            .WriteTo.File($@"C:\Users\xiaoyuz\source\repos\InferenceTuning\{level}_.txt", rollingInterval: RollingInterval.Day)
                            .CreateLogger();

            Log.Information("Start Experiment");
            IEnumerable<MathInput> trainSet = Directory.GetFiles("Math/train", "*.json", SearchOption.AllDirectories)
                .Select(file => JsonSerializer.Deserialize<MathInput>(File.ReadAllText(file))!);

            IEnumerable<MathInput> testSet = Directory.GetFiles("Math/test", "*.json", SearchOption.AllDirectories)
                .Select(file => JsonSerializer.Deserialize<MathInput>(File.ReadAllText(file))!);

            var key = "574924ad8a6c425eaa6f2fc9c45fc00e";
            var endpoint = "https://cog-avxpgc4w6gie6.openai.azure.com/";
            var gptClient = new OpenAIClient(new Uri(endpoint), new Azure.AzureKeyCredential(key));
            var defaultOption = new GPTInferenceOption
            {
                N = 5,
                Temperature = 1f,
                TopP = 1f,
            };
            var examples = CreateMathSolutionExample();
            var searchSpace = new SearchSpace<GPTInferenceOption>(defaultOption);
            var autoLLM = new AutoLLM(context);
            autoLLM.SetPipeline((context, examples, option) =>
            {
                return context.Transforms.CreateFewshotPromptTemplate(
                promptTempate: @"
Solve the question carefully. Simplify your answer as much as possible. Format your answer as json.
${Example}

question: ${problem}
response(in json):
",
                examplePromptTemplate: @"
###Example###
question: ${problem}
response(in json):
{
    ""solution"": ""${solution}"",
    ""simplifiedSolution"":""${simplifiedSolution}""
}
",
                inputColumnNames: new[] { "question" },
                outputColumnName: "prompt",
                exampleVariableName: "Example",
                examples.ToArray())
                .Append(context.Transforms.GPT3_5(gptClient, option));
            }, examples, searchSpace);

            Func<MLContext, IDataView, double> evaluator = (context, eval) =>
            {
                var responses = eval.GetColumn<string[]>("Responses");
                var actualSolutions = eval.GetColumn<string>("solution")
                    .Select(x => Regex.Match(x, @"\\boxed\{(.*)\}").Groups[1].Value)
                    .ToArray();
                var candidates = responses.Select(x =>
                {
                    var candidate = new List<string>();
                    foreach (var response in x)
                    {
                        try
                        {
                            var json = JsonSerializer.Deserialize<Dictionary<string, string>>(response);
                            if (json?.TryGetValue("simplifiedSolution", out var simplifiedSolution) is true)
                            {
                                candidate.Add(simplifiedSolution);
                            }
                        }
                        catch (Exception)
                        {
                            Log.Warning("Failed to parse response: {response}", response);
                        }
                    }

                    return candidate.ToArray();
                });

                var evaluationExamples = CreateEvaluationExample();
                var evaluationOption = new GPTInferenceOption
                {
                    N = 1,
                    Temperature = 0f,
                };

                var evaluationPipeline = context.Transforms.CreateFewshotPromptTemplate(
                promptTempate: @"
Determine if the most frequent candidate answer is correct.
Fist, determine the most frequent number in candidate answers. If there is a tie, don't pick any answer.
Then, determine if the most frequent number is the same as the actual answer. Just compare the number and ignore difference in measurement or format.
Finally, return true or false.
${Examples}

Candidate answer: ${candidates}
Actual answer: ${actual}
Response:",
                examplePromptTemplate: @"
###Example###
Candidate answer: ${candidates}
Actual answer: ${actual}
Resposne: ${response}
",
                inputColumnNames: new[] { "CandidateAnswer", "ActualAnswer" },
                outputColumnName: "prompt",
                exampleVariableName: "Examples",
                examples: evaluationExamples)
                .Append(context.Transforms.GPT3_5(gptClient, evaluationOption));

                var input = context.Data.LoadFromEnumerable(Enumerable.Zip(actualSolutions, candidates)
                     .Select(x => new MathInput
                     {
                         Actual = x.First,
                         Candidates = x.Second,
                     }));

                var evaluationResult = evaluationPipeline.Fit(input).Transform(input);

                var totolLine = responses.Count();
                var correctLine = evaluationResult.GetColumn<string[]>("Responses").Select(x => x.First())
                    .Where(x => x.ToLower().Contains("true")).Count();
                var response = evaluationResult.GetColumn<string[]>("Responses").Select(x => x.First()).ToArray();
                var tokens = evaluationResult.GetColumn<int>("Tokens");
                Log.Information($"Correct: {correctLine}/{totolLine}", correctLine, totolLine);
                Log.Information($"Raw Response: {string.Join("\n", response)}");
                Log.Information($"Tokens usage in total: {tokens.Sum()}");
                return (double)correctLine / totolLine;
            };

            autoLLM.SetEvaluator(evaluator);

            var model = autoLLM.Fit(context.Data.LoadFromEnumerable(trainSet.Take(20)), context.Data.LoadFromEnumerable(trainSet.Skip(10).Take(10)));
            var result = model.Transform(context.Data.LoadFromEnumerable(testSet.Take(20)));
            var score = evaluator(context, result);
            Log.Information($"Score on test dataset: {score}");
        }

        public static object[] CreateMathSolutionExample()
        {
            return new[]
            {
                new
                {
                    problem = "what is 1+1",
                    solution = "1 +  1 is \\boxed{2}",
                    simplifiedSolution = "2",
                },
                new
                {
                    problem = "what is 1+2",
                    solution = "1 +  2 is \\boxed{3}",
                    simplifiedSolution = "3",
                },
                new
                {
                    problem = "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?",
                    solution = "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes.",
                    simplifiedSolution = "2",
                },
                new
                {
                    problem = "If $2^8=4^x$, what is the value of $x$?",
                    solution = "Rewrite $4$ as $2^2$ to find $4^x=2^{2x}$.  Since $2^8=2^{2x}$, we have $2x=8$ which implies $x=\\boxed{4}$",
                    simplifiedSolution = "4",
                },
                new
                {
                    problem = "These two spinners are divided into thirds and quarters, respectively. If each of these spinners is spun once, what is the probability that the product of the results of the two spins will be an even number? Express your answer as a common fraction.\n\n[asy]\n\nsize(5cm,5cm);\n\ndraw(Circle((0,0),1));\n\ndraw(Circle((3,0),1));\n\ndraw((0,0)--(0,1));\n\ndraw((0,0)--(-0.9,-0.47));\n\ndraw((0,0)--(0.9,-0.47));\n\ndraw((2,0)--(4,0));\n\ndraw((3,1)--(3,-1));\n\nlabel(\"$3$\",(-0.5,0.3));\n\nlabel(\"$4$\",(0.5,0.3));\n\nlabel(\"$5$\",(0,-0.5));\n\nlabel(\"$5$\",(2.6,-0.4));\n\nlabel(\"$6$\",(2.6,0.4));\n\nlabel(\"$7$\",(3.4,0.4));\n\nlabel(\"$8$\",(3.4,-0.4));\n\ndraw((0,0)--(0.2,0.8),Arrow);\n\ndraw((3,0)--(3.2,0.8),Arrow);\n\n[/asy]",
                    solution = "We will subtract the probability that the product is odd from 1 to get the probability that the product is even. In order for the product to be odd, we must have both numbers be odd. There are $2\\cdot2=4$ possibilities for this (a 3 or 5 is spun on the left spinner and a 5 or 7 on the right) out of a total of $3\\cdot4=12$ possibilities, so the probability that the product is odd is $4/12=1/3$. The probability that the product is even is $1-1/3=\\boxed{\\frac{2}{3}}$.",
                    simplifiedSolution = "\\frac{2}{3}",
                },
                new
                {
                    problem = "Find the largest prime divisor of 11! + 12!",
                    solution = "Since $12! = 12 \\cdot 11!$, we can examine the sum better by factoring $11!$ out of both parts: $$ 11! + 12! = 11! + 12 \\cdot 11! = 11!(1 + 12) = 11! \\cdot 13. $$Since no prime greater than 11 divides $11!$, $\\boxed{13}$ is the largest prime factor of $11! + 12!$.",
                    simplifiedSolution = "13",
                },
            };
        }

        public static object[] CreateEvaluationExample()
        {
            return new[]
            {
                new
                {
                    candidates = new[]{"2", "2", "3"},
                    actual = "2",
                    response = "true, because 2 is the most frequent number in candidates and is equal to actual answer",
                },
                new
                {
                    candidates = new[]{"2", "2", "3"},
                    actual = "3",
                    response = "false, because 2 is the most frequent number in candidates but is not equal to actual answer",
                },
                new
                {
                    candidates = new[]{"2", "2", "3"},
                    actual = "4",
                    response = "false, because 2 is the most frequent number in candidates but is not equal to actual answer",
                },
                new
                {
                    candidates = new[]{"1.5", "3", "3/2", "\\frac{3}{2}"},
                    actual = "3/2",
                    response = "true, because 1.5 is the most frequent number in candidates and is equal to actual answer",
                },
            };
        }
    }
}
