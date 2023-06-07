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
