using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.SearchSpace;
using Serilog;
using System.Text.Json;

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
        private bool _inferenceTuning;

        public AutoLLM(MLContext context, bool inferenceTuning = true)
        {
            _context = context;
            _inferenceTuning = inferenceTuning;
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
            ITransformer? bestModel = null;

            for (int i = 0; i <= Math.Sqrt(exampleCount); ++i)
            {
                var examplesToPick = _random.Next(1, exampleCount);
                var pickedExamples = _examples.OrderBy(x => _random.Next()).Take(examplesToPick).ToArray();
                Log.Debug($"Select examples: {JsonSerializer.Serialize(pickedExamples, new JsonSerializerOptions
                {
                    WriteIndented = true,
                })}");
                var pipeline = _pipelineFunc(_context, pickedExamples, defaultParameter);
                var model = pipeline.Fit(trainSet);
                var eval = model.Transform(validationSet);
                var score = _evaluatorFunc(_context, eval);
                if (score > bestScore)
                {
                    bestScore = score;
                    selectedExamples = pickedExamples.ToList();
                    bestModel = model;
                }
                Log.Debug($"iteration: {i} Score: {score}");
            }

            Log.Information($"Best score after example selection: {bestScore}");
            Log.Information($"Best examples: {JsonSerializer.Serialize(selectedExamples)}");

            if (_inferenceTuning)
            {
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
            else
            {
                return bestModel!;
            }
        }
    }
}
