using System;
using System.IO;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace BinaryClassification_SentimentAnalysis
{

    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, "datasets", "data.tsv");
        private static string TestDataPath => Path.Combine(AppPath, "datasets", "test.tsv");

        static void Main(string[] args)
        {
            //1. Create ML.NET context/environment
            var env = new LocalEnvironment();

            //2. Create DataReader with data schema mapped to file's columns
            var reader = TextLoader.CreateReader(env, ctx => (label: ctx.LoadFloat(0),
                                                              text: ctx.LoadText(1)));

            //3. Create an estimator to use afterwards for creating/traing the model.
            var mctx = new MulticlassClassificationContext(env);
            var est = reader.MakeNewEstimator().Append(row =>
            {
                //TokenizeText().WordEmbeddings(@"C:\Users\timm\Downloads\wiki.nl.vec")
                var featurizedText = row.text.OneHotEncoding();  //Convert text to numeric vectors
                var prediction = mctx.Trainers.Sdca(row.label.ToKey(), featurizedText);  //Specify SDCA trainer based on the label and featurized text columns
                return (row.label, prediction);  //Return label and prediction columns. "prediction" holds predictedLabel, score and probability
            });
            //4. Build and train the model

            //Load training data
            var traindata = reader.Read(new MultiFileSource(TrainDataPath));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = est.Fit(traindata);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            //5. Evaluate the model

            //Load test data
            var testdata = reader.Read(new MultiFileSource(TestDataPath));
            //Console.WriteLine("=============== Evaluating Model's accuracy with Test data===============");
            var predictions = model.Transform(testdata);
            Console.WriteLine(predictions);
            var metrics = mctx.Evaluate(predictions, row => row.Item1.ToKey(), row => row.Item2);
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.TopKAccuracy:P2}");
            Console.WriteLine(metrics);

            //6. Test Sentiment Prediction with one sample text 
            var predictionFunct = model.AsDynamic.MakePredictionFunction<SentimentIssue, SentimentPrediction>(env);

            SentimentIssue sampleStatement = new SentimentIssue
            {
                text = "permanente vorming in oncologische zorg"
            };

            var resultprediction = predictionFunct.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Test of model with a sample ===============");
            Console.WriteLine($"Text: {sampleStatement.text} | Prediction: {(resultprediction.PredictionLabel)}");

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

    }
}
namespace BinaryClassification_SentimentAnalysis
{
    public class SentimentIssue
    {
        [Column(ordinal: "0")]
        public string label { get; set; }
        [Column(ordinal: "1")]
        public string text { get; set; }
    }
}
namespace BinaryClassification_SentimentAnalysis
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public Key<uint, string> PredictionLabel { get; set; }

        //[ColumnName("prediction.probability")]
        //public float Probability { get; set; }

        //[ColumnName("prediction.score")]
        //public float Score { get; set; }
    }
}