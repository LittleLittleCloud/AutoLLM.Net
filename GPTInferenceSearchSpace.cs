using Microsoft.ML.SearchSpace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferenceTuning
{
    public class GPTInferenceOption
    {
        public string PromptTemplate { get; set; }

        [Range(0f, 2f, 1f, false)]
        public float Temperature { get; set; }

        [Range(0f, 1f, 1f, false)]
        public float TopP { get; set; }

        public int MaxTokens { get; set; } = 1024;

        [Range(1, 100, 5, false)]
        public int N { get; set; }
    }
}
