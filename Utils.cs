using Azure.AI.OpenAI;
using Serilog;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;

namespace InferenceTuning
{
    public static class Utils
    {
        public static async Task<bool> IsCorrect(OpenAIClient client, string[] candidates, string expected)
        {
            // uss gpt model to determine if answer with the most vote is correct
            var validateOption = new ChatCompletionsOptions
            {
                ChoicesPerPrompt = 1,
                Temperature = 0,
                NucleusSamplingFactor = 0,
                MaxTokens = 1024,
            };

            // log answer
            Log.Information($"candidate answers: {string.Join(", ", candidates)}");
            Log.Information($"actual answer: {expected}");

            var prompt = $$"""
                    Determine if the most frequent candidate answer is correct.
                    Fist, determine the most frequent number in candidate answers. If there is a tie, don't pick any answer.
                    Then, determine if the most frequent number is the same as the actual answer. Just compare the number and ignore difference in measurement or format.
                    Finally, return true or false.
                    ### Examples 1
                    Candidate answer:
                    2, 3, 2, 2, 2
                    Actual answer:
                    2
                    Most frequent number:
                    2
                    Return:
                    true
                    ###
                    ### Examples 2
                    Candidate answer:
                    1.5, 3, 3/2, \frac{3}{2}
                    Actual answer:
                    3/2
                    Most frequent number:
                    1.5
                    Return:
                    true
                    ###
                    ### Examples 3
                    Candidate answer:
                    1,2,3,4,5
                    Actual answer:
                    5
                    Most frequent number:
                    (there's tie)
                    Return:
                    false (no most frequent number)
                    ###
                    ### Examples 4
                    Candidate answer:
                    8, -8, 0, -\frac{7}{2}, 1
                    Actual answer:
                    0
                    Most frequent number:
                    (there's tie)
                    Return:
                    false (0 is not the most frequent number)
                    ###
                    ### Examples 5
                    Cadidate answer:
                    1, 2, 3, 4, 5
                    Actual answer:
                    6
                    Most frequent number:
                    (there's tie)
                    Return:
                    false (6 is not in the candidate answer)
                    ###
                    Candidate answer:
                    {{string.Join(",", candidates)}}
                    Actual answer:
                    {{expected}}
                    """;

            validateOption.Messages.Add(new ChatMessage(ChatRole.System, prompt));
            var validateResponse = await client.GetChatCompletionsAsync("chat", validateOption);
            var validateAnswer = validateResponse.Value.Choices.First()?.Message.Content;
            Log.Information($"validate answer: {validateAnswer}");

            if (validateAnswer?.Trim().ToLower().Contains("true") is true)
            {
                Log.Information("correct");
                return true;
            }

            Log.Information("incorrect");
            return false;
        }

        public static string RenderTemplate<TInput>(string template, TInput input)
        {
            // step 1
            // serialze to json object
            var json = JsonSerializer.Serialize(input);
            var jsonNode = JsonNode.Parse(json);

            // step 2
            // replace ${variable} with jsonNode["variable"]

            // step 2.1
            // find all variables
            var variables = new List<string>();
            var regex = new Regex(@"\$\{(\w+)\}");
            var matches = regex.Matches(template);
            foreach (Match match in matches)
            {
                variables.Add(match.Groups[1].Value);
            }

            foreach (var variable in variables)
            {
                if(jsonNode![variable] != null)
                {
                    // step 2.2
                    // replace ${variable} with jsonNode["variable"]
                    var variableRegex = new Regex($@"\$\{{{variable}\}}");
                    template = variableRegex.Replace(template, jsonNode[variable]!.ToString());
                }
                
            }

            return template;
        }
    }
}
