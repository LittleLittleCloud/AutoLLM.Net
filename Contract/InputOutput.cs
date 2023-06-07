using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InferenceTuning.Contract
{
    public class GPTInput
    {
        public string Prompt { get; set; }
    }

    public class GPTOutput
    {
        public string[] Responses { get; set; }

        public int Tokens { get; set; }
    }
}
