using Microsoft.ML.SearchSpace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferenceTuning
{
    public class GPTInferenceSearchSpace
    {
        [Range(0.3f, 1f, false)]
        public float Temperature { get; set; }
    }
}
