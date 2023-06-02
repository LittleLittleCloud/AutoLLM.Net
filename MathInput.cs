using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace InferenceTuning
{
    public class MathInput
    {
        [ColumnName("problem")]
        [JsonPropertyName("problem")]
        public string Problem { get; set; }

        [ColumnName("level")]
        [JsonPropertyName("level")]
        public string Level { get; set; }

        [ColumnName("type")]
        [JsonPropertyName("type")]
        public string Type { get; set; }

        [ColumnName("solution")]
        [JsonPropertyName("solution")]
        public string Solution { get; set; }

        [ColumnName("actual")]
        [JsonPropertyName("actual")]
        public string Actual { get; set; }

        [ColumnName("candidates")]
        [JsonPropertyName("candidates")]
        public string[] Candidates { get; set; }
    }

    public class MathOutput : MathInput
    {
        public string PredictedSolution { get; set; }

        public string[] Responses { get; set; }
    }
}
