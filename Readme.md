# AutoLLM - Automatic Inference Parameter Tuning and example selection for LLM

## Inference parameter tuning
[This Research](https://arxiv.org/abs/2303.04673) shows inference parameter is critical for the LLM generative capability. Providing a search space for inferencing and let `AutoLLM` finds the most suitable parameter.

## Example selection
Fewshot can help LLM generate better response in most of situation. And `AutoLLM` can determine the most suitable examples combination for you.

## Using Example
```csharp
var context = new MLContext();
var autoLLM = context.Auto().CreateAutoLLM();
var examples = new[]{
    new
    {
        problem = "what answer is 1 + 1",
        answer = "2",
        reason = "1 + 1 is 2",
    },
    new
    {
        problem = "If $2^8=4^x$, what is the value of $x$?",
        answer = "4",
        reason = "Rewrite $4$ as $2^2$ to find $4^x=2^{2x}$.  Since $2^8=2^{2x}$, we have 2x=8$ which implies $x=\\boxed{4}$",
    },
    // other examples
};

// temperature: [0.3, 2]
// N: [3, 100]
// ... other search space
GPTSearchSpace searchSpace;

// train validation dataset
IDataView train, validation;

autoLLM.SetTrialRunner((context, examples, option, train, validation) =>{
        var pipeline = context.Transform.CreateFewshotPromptTemplate(
            promptTemplate: @"
            Solve the question carefully, Simplify your answer as much as possible.
            ${Example}

            question: ${problem}
            response(in json):",
            examplePromptTemplate: @"
            ### Example ###
            question: ${problem}
            response(in json):
            {
                ""answer"": ${answer},
                ""reason"": ${reason}
            }",
            outputColumnName: "prompt",
            exampleVariableName: "Example",
            examples)
            .Append(context.Transforms.GPT3_5(
                inputColumnName: "prompt",
                outputColumnName: "response",
                temperature: option.Temperature,
                N: option.N,
                // other options
                apiKey: Environment.GetVariable("api-key"),
            ));
        var model = pipeline.Fit(train);
        var eval = model.Transform(validation);
        // calculate score

        return new TrialResult{
            metric = score,
            loss = -score,
            model = model,
            parameter = option,
            examples = examples,
        };
    )
}, examples, searchSpace);

var bestModel = autoLLM.Fit(train, validation);

// use model
var input = new {
    problem = "what's 2 + 3",
};

var output = bestModel.Transform(context.Data.LoadFromEnumerable(input))
var response = output.GetColumn<string>("response").First();

response
/*
{
    "answer": "5",
    "reason": "because 2 + 3 = 5"
}
*/
```

## Case study
### [Math](https://arxiv.org/abs/2103.03874)
Use LLM to resolve math problems. Input is problem and output is answer inside \\box{}. Evaluate metric is accuracy. Code is [here](./Experiment/MathExperiment.cs)
#### Settings
- LLM Model: GPT3.5 turbo
- number of example candidates: 6
- example selector: random
- number of train dataset: 20
- number of validation dataset: 10
- number of test dataset: 200

#### Result
| Level | oneshot + default inference parameter | oneshot + inference parameter tuning | all examples + default inference parameter | random example selector+ inference tuning | random example selector + default inference parameter |
|-------|--------------------------------------|---------------------------------------|--------------------------------------------|-----------------------------------------|-------------------------------------------------------|
| 2 | 0.885 | 0.94 | 0.985 | 0.985 | 0.885 |
| 3 | 0.78 | 0.9 | 0.935 | 0.77 | 0.78 |
| 4 | 0.685 | 0.77 | 0.94 | 0.93 | 0.7 |
| 5 | 0.465 | 0.495 | 0.91 | 0.94 | 0.455 |


## Reference
- [Flaml.AutoGen](https://microsoft.github.io/FLAML/docs/Use-Cases/Auto-Generation/)
- [Cost-Effective Hyperparameter Optimization for Large Language Model Generation Inference](https://arxiv.org/abs/2303.04673)
- [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)
