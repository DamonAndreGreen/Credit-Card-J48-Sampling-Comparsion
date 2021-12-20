// Importing the proper libraries and WEKA
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.supervised.instance.Resample; 
import weka.filters.supervised.instance.SpreadSubsample;


public class Fraud {

	public static void main(String args[])
    {
  
        try {
  
            // Creating J48 classifier for the  tree
            J48 j48Classifier = new J48();
  
            // Setting the path for the dataset
            String FraudDataset = "Path directory to CreditCard.arff";
            BufferedReader bufferedReader
            = new BufferedReader(
                new FileReader(FraudDataset));
            
            SpreadSubsample spread = new SpreadSubsample();
           spread.setOptions(Utils.splitOptions("-M 1.0"));
           ///  meta-classifier
            FilteredClassifier fc = new FilteredClassifier();
            fc.setFilter(spread);
            fc.setClassifier(j48Classifier);
            
           Resample resam = new Resample();
            resam.setOptions(Utils.splitOptions("-B 1.0 -Z 130.3"));
            // meta-classifier
            FilteredClassifier rc = new FilteredClassifier();
             rc.setFilter(resam);
             rc.setClassifier(j48Classifier);

        // Creating the data set instances
        Instances datasetInstances
            = new Instances(bufferedReader);

  
        datasetInstances.setClassIndex(
            datasetInstances.numAttributes() - 1);

        Evaluation evaluation
            = new Evaluation(datasetInstances);

        // Cross Validate Model. 10 Folds
        evaluation.crossValidateModel(
            j48Classifier, datasetInstances, 10,
            new Random(1));
        System.out.println(evaluation.toSummaryString(
            "\n J48 Regular Results", false));
        
        evaluation.crossValidateModel(
                fc, datasetInstances, 10,
                new Random(1));
            System.out.println(evaluation.toSummaryString(
               "\nJ48 Undersampling Results", false));
            
           evaluation.crossValidateModel(
                   rc, datasetInstances, 10,
                   new Random(1));
               System.out.println(evaluation.toSummaryString(
                    "\nJ48 Oversampling Results", false));
            
        
    }

    // Catching exceptions
    catch (Exception e) {
        System.out.println("Error Occured!!!! \n"
                           + e.getMessage());
    }


    System.out.print("J48 comparasion successfully executed.");
}
	
}
