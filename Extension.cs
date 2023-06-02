using Azure.AI.OpenAI;
using InferenceTuning.Contract;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace InferenceTuning
{
    public static class Extension
    {
        public static IEstimator<ITransformer> CreateFewshotPromptTemplate<TExample>(
            this TransformsCatalog transforms,
            string promptTempate,
            string examplePromptTemplate,
            string[] inputColumnNames,
            string outputColumnName,
            string exampleVariableName,
            TExample[] examples)
        {
            //step 1
            // render example prompt
            var examplePrompts = examples.Select(example =>
            {
                var examplePrompt = Utils.RenderTemplate(examplePromptTemplate, example);
                return examplePrompt;
            }).ToArray();

            return transforms.CustomMapping<MathInput, GPTInput>(
                (input, output) =>
                {
                    //step 2
                    // render prompt
                    var prompt = Utils.RenderTemplate(promptTempate, new Dictionary<string, string>
                    {
                        { exampleVariableName, string.Join("\n", examplePrompts)},
                    });
                    prompt = Utils.RenderTemplate(prompt, input);
                    output.Prompt = prompt;
                },
                contractName: "CreateFewshotPromptTemplate");
        }
        
        public static IEstimator<ITransformer> GPT3_5(this TransformsCatalog transform, OpenAIClient client, GPTInferenceOption gptOption)
        {
            return transform.CustomMapping<GPTInput, GPTOutput>(
                          (input, output) =>
                          {
                              var chatMessages = new[]
                              {
                                new ChatMessage(ChatRole.System, input.Prompt),
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
                              var response = client.GetChatCompletions("chat", option);
                              var answers = response.Value.Choices.Select(choice => choice.Message.Content).ToList();
                              output.Responses = answers.ToArray();
                              output.Tokens = response.Value.Usage.TotalTokens;
                          }, contractName: "GPT3_5");
        }
    }
}
