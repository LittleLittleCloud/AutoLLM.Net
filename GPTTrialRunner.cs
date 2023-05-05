using Azure.AI.OpenAI;
using Microsoft.ML.AutoML;
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
            var gptOption = settings.Parameter["gpt"].AsType<GPTInferenceSearchSpace>();
            var outputSolution = new List<string>();
            var correctCount = 0;
            var totalCount = 0;
            foreach (var input in trainSet)
            {
                var chatMessages = new[]
                {
                    new ChatMessage(ChatRole.System, "Solve the following math problem, include thought process and wrap the answer in \\boxed{}"),
                    new ChatMessage(ChatRole.User, input.Problem),
                };
                var option = new ChatCompletionsOptions()
                {
                    ChoicesPerPrompt = 5,
                    Temperature = gptOption.Temperature,
                };
                foreach (var chatMessage in chatMessages)
                {
                    option.Messages.Add(chatMessage);
                }
                var response = await openAIClient.GetChatCompletionsAsync("chat", option);

                var answers = response.Value.Choices.Select(choice => choice.Message.Content).ToList();

                // print answers
                var answerInBox = answers.Select(a => Regex.Match(a, @"\\boxed\{(.*)\}").Groups[1].Value).ToList();
                Console.WriteLine($"answers: {string.Join(", ", answerInBox)}");
                var answerWithTheMostVote = answerInBox.GroupBy(answer => answer).OrderByDescending(group => group.Count()).First().Key;

                // use regex to extract answer from \\boxed{}
                // example: blablabla \\boxed{answer} blablabla
                var correctAnswer = Regex.Match(input.Solution, @"\\boxed\{(.*)\}").Groups[1].Value;

                Console.WriteLine($"predict answer: {answerWithTheMostVote}, correct answer: {correctAnswer}");

                if(correctAnswer == answerWithTheMostVote)
                {
                    correctCount++;
                }

                totalCount++;
            }

            var accuracy = correctCount / (double)totalCount;

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
