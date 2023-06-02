using Azure.AI.OpenAI;
using Microsoft.ML.AutoML;
using Serilog;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;

namespace InferenceTuning
{
    public class GPTTrialRunner : ITrialRunner
    {
        private readonly IEnumerable<MathInput> trainSet;
        private readonly OpenAIClient openAIClient;

        public GPTTrialRunner(IEnumerable<MathInput> trainSet, OpenAIClient openAIClient)
        {
            this.trainSet = trainSet;
            this.openAIClient = openAIClient;
        }

        public void Dispose()
        {
        }

        public async Task<TrialResult> RunAsync(TrialSettings settings, CancellationToken ct)
        {
            Log.Information($"Run trial {settings.TrialId}");
            var gptOption = settings.Parameter["gpt"].AsType<GPTInferenceOption>();
            Log.Information("Trial parameter:");
            Log.Information($"Temperature: {gptOption.Temperature}");
            Log.Information($"TopP: {gptOption.TopP}");
            Log.Information($"MaxTokens: {gptOption.MaxTokens}");
            Log.Information($"N: {gptOption.N}");
            var outputSolution = new List<string>();
            var correctCount = 0;
            var totalCount = 0;
            foreach (var input in trainSet)
            {
                var chatMessages = new[]
                {
                    new ChatMessage(ChatRole.System, string.Format(gptOption.PromptTemplate, input.Problem)),
                };
                var option = new ChatCompletionsOptions()
                {
                    ChoicesPerPrompt = gptOption.N,
                    Temperature = gptOption.Temperature,
                    NucleusSamplingFactor = gptOption.TopP,
                    MaxTokens = gptOption.MaxTokens,
                };
                foreach (var chatMessage in chatMessages)
                {
                    option.Messages.Add(chatMessage);
                }
                var response = await openAIClient.GetChatCompletionsAsync("chat", option);

                var answers = response.Value.Choices.Select(choice => choice.Message.Content).ToList();
                // print answers
                var answerInBox = answers.Select(a => Regex.Match(a, @"\\boxed\{(.*)\}").Groups[1].Value).ToArray();
                var correctAnswer = Regex.Match(input.Solution, @"\\boxed\{(.*)\}").Groups[1].Value;
                var isCorrect = await Utils.IsCorrect(openAIClient, answerInBox, correctAnswer);

                if(isCorrect)
                {
                    correctCount++;
                }

                totalCount++;
            }

            var accuracy = correctCount / (double)totalCount;

            Log.Information($"Trial {settings.TrialId} accuracy: {accuracy}");
            return new TrialResult()
            {
                Metric = accuracy,
                TrialSettings = settings,
                DurationInMilliseconds = 0,
                Loss = -accuracy,
            };
        }
    }
}
