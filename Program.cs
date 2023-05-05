// load Math dataset
using Azure.AI.OpenAI;
using InferenceTuning;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using System.Text.Json;

IEnumerable<MathInput> trainSet = Directory.GetFiles("Math/train", "*.json", SearchOption.AllDirectories)
    .Select(file => JsonSerializer.Deserialize<MathInput>(File.ReadAllText(file))!);

IEnumerable<MathInput> testSet = Directory.GetFiles("Math/test", "*.json", SearchOption.AllDirectories)
    .Select(file => JsonSerializer.Deserialize<MathInput>(File.ReadAllText(file))!);

var context = new MLContext();
var searchSpace = new SearchSpace<GPTInferenceSearchSpace>();
var key = "your key";
var endpoint = "your endpoint";
var gptClient = new OpenAIClient(new Uri(endpoint), new Azure.AzureKeyCredential(key));
var gptTrialRunner = new GPTTrialRunner(trainSet.Take(20), gptClient);

var experiment = context.Auto().CreateExperiment();
experiment.SetTrialRunner(gptTrialRunner)
          .AddSearchSpace("gpt", searchSpace)
          .SetCostFrugalTuner();

var result = await experiment.RunAsync();
var bestInferenceOption = result.TrialSettings.Parameter["gpt"].AsType<GPTInferenceSearchSpace>();

// use best inference option to evaluate test set
var correctCount = 0;
var totalCount = 0;
foreach(var input in testSet)
{
    var chatMessages = new[]
    {
        new ChatMessage(ChatRole.System, "Solve the following math problem:"),
        new ChatMessage(ChatRole.User, input.Problem),
    };
    var option = new ChatCompletionsOptions()
    {
        Temperature = bestInferenceOption.Temperature,
    };
    foreach (var chatMessage in chatMessages)
    {
        option.Messages.Add(chatMessage);
    }
    var response = await gptClient.GetChatCompletionsAsync("chat", option);

    if(input.Solution == response.Value.Choices.First().Message.Content)
    {
        correctCount++;
    }
    totalCount++;
}

Console.WriteLine($"Accuracy: {correctCount / (double)totalCount}");
