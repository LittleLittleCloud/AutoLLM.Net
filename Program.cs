// load Math dataset
using Azure.AI.OpenAI;
using InferenceTuning;
using InferenceTuning.Experiment;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using Serilog;
using System.Text.Json;
using System.Text.RegularExpressions;

await MathExperiment.RunAsync();
return;
