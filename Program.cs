// load Math dataset
using Azure.AI.OpenAI;
using InferenceTuning;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using Serilog;
using System.Text.Json;
using System.Text.RegularExpressions;

await AutoLLM.Example1();
return;
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

var context = new MLContext();
var defaultOption = new GPTInferenceOption
{
    N = 5,
    PromptTemplate = "{0} Solve the problem carefully. Simplify your answer as much as possible. Put the final answer in \\boxed{{}}.",
    Temperature = 1f,
    TopP = 1f,
};

var searchSpace = new SearchSpace<GPTInferenceOption>(defaultOption);
var key = "use your own key";
var endpoint = "https://cog-kqhfb2cpjkxnc.openai.azure.com/";
var gptClient = new OpenAIClient(new Uri(endpoint), new Azure.AzureKeyCredential(key));
var gptTrialRunner = new GPTTrialRunner(trainSet.Where(l => l.Level == level).Take(20), gptClient);

var experiment = context.Auto().CreateExperiment();
experiment.SetTrialRunner(gptTrialRunner)
          .AddSearchSpace("gpt", searchSpace)
          .SetCostFrugalTuner()
          .SetMaxModelToExplore(20);

var result = await experiment.RunAsync();
var bestInferenceOption = result.TrialSettings.Parameter["gpt"].AsType<GPTInferenceOption>();

// log best inference option
Log.Information($"Best inference option:");
Log.Information($"Temperature: {bestInferenceOption.Temperature}");
Log.Information($"TopP: {bestInferenceOption.TopP}");
Log.Information($"MaxTokens: {bestInferenceOption.MaxTokens}");
Log.Information($"N: {bestInferenceOption.N}");

Log.Information($"# of test on level 2: {testSet.Count()}");
foreach(var input in testSet.Where(t => t.Level == level))
{
    var chatMessages = new[]
    {
        new ChatMessage(ChatRole.System, string.Format(bestInferenceOption.PromptTemplate, input.Problem)),
    };

    var option = new ChatCompletionsOptions()
    {
        Temperature = bestInferenceOption.Temperature,
        NucleusSamplingFactor = bestInferenceOption.TopP,
        MaxTokens = bestInferenceOption.MaxTokens,
        ChoicesPerPrompt = bestInferenceOption.N,
    };
    var defaultInferenceOption = new ChatCompletionsOptions
    {
        Temperature = defaultOption.Temperature,
        NucleusSamplingFactor = defaultOption.TopP,
        MaxTokens = defaultOption.MaxTokens,
        ChoicesPerPrompt = defaultOption.N,
    };

    foreach (var chatMessage in chatMessages)
    {
        option.Messages.Add(chatMessage);
        defaultInferenceOption.Messages.Add(chatMessage);
    }

    var response = await gptClient.GetChatCompletionsAsync("chat", option);
    var correctAnswer = Regex.Match(input.Solution, @"\\boxed\{(.*)\}").Groups[1].Value;
    var defaultResponse = await gptClient.GetChatCompletionsAsync("chat", defaultInferenceOption);
    var candidateAnswers = response.Value.Choices.Select(choice => choice.Message.Content ?? string.Empty)
        .Select(content => Regex.Match(content, @"\\boxed\{(.*)\}").Groups[1].Value)
        .ToArray();
    var defaultCandidateAnswers = defaultResponse.Value.Choices.Select(choice => choice.Message.Content ?? string.Empty)
        .Select(content => Regex.Match(content, @"\\boxed\{(.*)\}").Groups[1].Value)
        .ToArray();

    Log.Information("check if best inference option is correct");
    var ifCorrect = await Utils.IsCorrect(gptClient, candidateAnswers, correctAnswer);
    Log.Information($"{ifCorrect}");
    Log.Information("check if default inference option is correct");
    var defaultIfCorrect = await Utils.IsCorrect(gptClient, defaultCandidateAnswers, correctAnswer);
    Log.Information($"{defaultIfCorrect}");

    // log meta information
    Log.Information($"Level={input.Level};" +
        $"Type={input.Type};" +
        $"Problem={input.Problem};" +
        $"Answer={correctAnswer};" +
        $"CandidateAnswersFromTuning={string.Join(",", candidateAnswers)};" +
        $"IsCandidateAnswerCorrect={ifCorrect};" +
        $"CandidateAnswersFromDefault={string.Join(",", defaultCandidateAnswers)};" +
        $"IsCandidateDefaultAnswerCorrect={defaultIfCorrect}");
}

