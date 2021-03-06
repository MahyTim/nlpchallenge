﻿using System;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Training;
using Microsoft.ML;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.TextAnalytics;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace BinaryClassification_SentimentAnalysis
{
    internal static class Program
    {

        static void Main(string[] args)
        {
            var testDataFile = TestData.PrepareTestDataAndReturnPath(23,61,72);

            for (int i = 0; i < 5; i++)
            {
                var env = new LocalEnvironment();
                var classification = new MulticlassClassificationContext(env);
                var reader = TextLoader.CreateReader(env, ctx => (Label: ctx.LoadText(0), Text: ctx.LoadText(1)), separator: ',', hasHeader: false);
                var data = reader.Read(new MultiFileSource(testDataFile));

                var learningPipeline = reader.MakeNewEstimator()
                    .Append(r => (Label: r.Label.ToKey(), Features: r.Text.FeaturizeText(advancedSettings: s =>
                    {
                        s.KeepDiacritics = false;
                        //s.KeepNumbers = false; bv PC
                        s.KeepPunctuations = false;
                        s.TextCase = TextNormalizerTransform.CaseNormalizationMode.Lower;
                        s.TextLanguage = TextTransform.Language.Dutch;
                        s.VectorNormalizer = TextTransform.TextNormKind.LInf;
                    })))
                    .Append(r => (Label: r.Label, Predications: classification.Trainers.Sdca(r.Label, r.Features)));

                var (trainData, testData) = classification.TrainTestSplit(data, testFraction: 0.2);
                var model = learningPipeline.Fit(trainData);
                var metrics = classification.Evaluate(model.Transform(testData), r => r.Label, r => r.Predications);
                Console.WriteLine(metrics.AccuracyMicro);
                Console.WriteLine(metrics.AccuracyMacro);

                //var cvResults = classification.CrossValidate(data, learningPipeline, r => r.Label, numFolds: 5);
                //var microAccuracies = cvResults.Select(r => r.metrics.AccuracyMicro);
                //Console.WriteLine(microAccuracies.Average());
                //var macroAccuracies = cvResults.Select(r => r.metrics.AccuracyMacro);
                //Console.WriteLine(macroAccuracies.Average());
                Console.WriteLine("-----------");
            }
        }


    }
}
