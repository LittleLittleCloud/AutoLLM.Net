using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.SearchSpace;
using Serilog;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace InferenceTuning
{
    public class AutoLLMInferenceParameterTrialRunner : ITrialRunner
    {
        private Func<MLContext, IDataView, double> _evaluatorFunc;
        private Func<MLContext, Parameter, IEstimator<ITransformer>> _pipelineFunc;
        private MLContext _mlContext;
        private IDataView _trainData;
        private IDataView _validationData;

        public AutoLLMInferenceParameterTrialRunner(MLContext mlContext, IDataView trainData, IDataView validationData, Func<MLContext, IDataView, double> evaluatorFunc, Func<MLContext, Parameter, IEstimator<ITransformer>> pipelineFunc)
        {
            _mlContext = mlContext;
            _trainData = trainData;
            _validationData = validationData;
            _evaluatorFunc = evaluatorFunc;
            _pipelineFunc = pipelineFunc;
        }

        public void Dispose()
        {
            return;
        }

        public Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct)
        {
            var parameter = settings.Parameter["llm"];

            var pipeline = _pipelineFunc(_mlContext, parameter);
            var model = pipeline.Fit(_trainData);
            var eval = model.Transform(_validationData);
            var metric = _evaluatorFunc(_mlContext, eval);
            Log.Information($"Trial {settings.TrialId} parameter: {JsonSerializer.Serialize(parameter, new JsonSerializerOptions
            {
                WriteIndented = true,
            })}");

            Log.Information($"Trial {settings.TrialId} metric: {metric}");
            return Task.FromResult(new TrialResult()
            {
                Metric = metric,
                Loss = -metric,
                TrialSettings = settings,
                Model = model,
            });
        }
    }
}
