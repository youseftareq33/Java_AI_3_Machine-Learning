
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Statistics;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;
import weka.core.Utils;


import java.io.File;
import java.util.Arrays;


public class Driver_Test {

    public static void main(String[] args) throws Exception{
    	System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
    	System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");
    	System.setProperty("com.github.fommil.netlib.ARPACK", "com.github.fommil.netlib.F2jARPACK");
    	
    	DataSource source = new DataSource("Height_Weight.arff"); // load path of datasource from arff file
    	Instances data = source.getDataSet(); // get data from file
    	System.out.println(data.toSummaryString()); // print summary
    	System.out.println("-----------------------------------------------");
    	
    	//--------------------------------------------------------------------------- 
    	// 1) 
    	
	    	// * convert the height from inches to CMS 
	    	Attribute heightAttribute = data.attribute("Height"); // set heightAttribute
	        
	        for (int i = 0; i < data.size(); i++) {
	            double heightInches = data.instance(i).value(heightAttribute); // get height instance value 
	
	            double heightCms = heightInches * 2.54; // convert it to cms
	            data.instance(i).setValue(heightAttribute, heightCms); // set the new value of cms
	        }
	        
	        // * convert the weight from pounds to kilograms 
	        Attribute weightAttribute = data.attribute("Weight"); // set weightAttribute
	        
	        for (int i = 0; i < data.size(); i++) {
	            double weightPounds = data.instance(i).value(weightAttribute); // get weight instance value 
	
	            double weightKgs = weightPounds * 0.453592; // convert it to kilograms
	            data.instance(i).setValue(weightAttribute, weightKgs); // set the new value of kilograms
	        }
	        
	     //---------------------------------------------------------------------------  
	     // 2)        

	        System.out.println("2)"+"\n");
	        int numOfAttributes=data.numAttributes(); // number of attributes
	        
	        for (int i=0; i<numOfAttributes; i++) {
	        	// for height
	        	if(data.attribute(i).equals(heightAttribute)) {
	        		System.out.println(" - Attribute " + (i+1) + ": " + data.attribute(i).name());
	        		AttributeStats as = data.attributeStats(i); // use attributeStats to get stats
	        		Stats s=as.numericStats; // set numericStates to get the main statistics
	        		
	        		double[] values = data.attributeToDoubleArray(i); // to get Median
                    Arrays.sort(values);

                    
	        		System.out.println(" Mean: "+s.mean+", Median: "+calculateMedian(values)+", Standard Deviation: "+s.stdDev+
	        				           ", Min value: "+s.min+", Max value: " + s.max);
	        	}
	        	// for weight
	        	else if(data.attribute(i).equals(weightAttribute)) {
	        		System.out.println(" - Attribute " + (i+1) + ": " + data.attribute(i).name());
	        		AttributeStats as = data.attributeStats(i); // use attributeStats to get stats
	        		Stats s=as.numericStats; // set numericStates to get the main statistics
	        		
	        		double[] values = data.attributeToDoubleArray(i); // to get Median
                    Arrays.sort(values);
                    
                    
                    System.out.println(" Mean: "+s.mean+", Median: "+calculateMedian(values)+", Standard Deviation: "+s.stdDev+
                    		           ", Min value: "+s.min+", Max value: " + s.max);
	        	}

            }
	        
	        System.out.println("\n"+"-----------------------------------------------");
	      //---------------------------------------------------------------------------  
		  // 3)
	        
	       data.setClassIndex(data.numAttributes()-1); // or you can use this code: data.setClass(data.attribute("Weight"));
	       
	       int trainSize = (int)(data.numInstances() * 0.7); // size of training data
	       int testSize = data.numInstances() - trainSize; // size of test data

	        Instances trainingData = new Instances(data, 0, trainSize); //trainingData: 0 start ... till train size 70
	        Instances testData = new Instances(data, trainSize, testSize); //testData: 70 start ... till 100
	
	        
	      //---------------------------------------------------------------------------  
		  // 4)  
	        
	        System.out.println("4)"+"\n");
	        data.randomize(new java.util.Random()); // make random to the data
	        Instances dataM1 = new Instances(trainingData, 0, 100); // data of Model 1 (Select a subset of 100 instances)
	        
	
	        // Generate the linear regression model (M1)
	        LinearRegression m1 = new LinearRegression();
	        m1.buildClassifier(dataM1); // build the relation
	        System.out.println("Relation M1: ");
	        System.out.print(m1+"\n");
	
	        Evaluation e_m1 = new Evaluation(trainingData); // set the training data on evaluation model
	        e_m1.evaluateModel(m1, testData); // evaluate the model on the test set
	        
	        // Print regression metrics
	        System.out.println("\n"+"Regression Metrics for Model (M1):"+"\n");
	        System.out.println("Mean Absolute Error (MAE): " + e_m1.meanAbsoluteError());
	        System.out.println("Root Mean Squared Error (RMSE): " + e_m1.rootMeanSquaredError());
	        System.out.println("Relative Absolute Error (RAE): " + e_m1.relativeAbsoluteError());
	        System.out.println("Correlation coefficient: " + e_m1.correlationCoefficient());
	        System.out.println("\n"+"-----------------------------------------------");
	      //---------------------------------------------------------------------------  
		  // 5)  
	        
	        System.out.println("5)"+"\n");
	        data.randomize(new java.util.Random()); // make random to the data
	        Instances dataM2 = new Instances(trainingData, 0, 1000); // data of Model 2 (Select a subset of 1000 instances)
	
	        // Generate the linear regression model (M2)
	        LinearRegression m2 = new LinearRegression();
	        m2.buildClassifier(dataM2); // build the relation
	        System.out.println("Relation M2: ");
	        System.out.print(m2+"\n");
	
	        Evaluation e_m2 = new Evaluation(trainingData); // set the training data on evaluation model
	        e_m2.evaluateModel(m2, testData); // evaluate the model on the test set
	        
	        // Print regression metrics
	        System.out.println("\n"+"Regression Metrics for Model (M2):"+"\n");
	        System.out.println("Mean Absolute Error (MAE): " + e_m2.meanAbsoluteError());
	        System.out.println("Root Mean Squared Error (RMSE): " + e_m2.rootMeanSquaredError());
	        System.out.println("Relative Absolute Error (RAE): " + e_m2.relativeAbsoluteError());
	        System.out.println("Correlation coefficient: " + e_m2.correlationCoefficient());
	        System.out.println("\n"+"-----------------------------------------------");
	      //---------------------------------------------------------------------------  
	      // 6)  
	        System.out.println("6)"+"\n");
	        	data.randomize(new java.util.Random()); // make random to the data
		        Instances dataM3 = new Instances(trainingData, 0, 5000); // data of Model 3 (Select a subset of 5000 instances)
		
		        // Generate the linear regression model (M3)
		        LinearRegression m3 = new LinearRegression();
		        m3.buildClassifier(dataM3); // build the relation
		        System.out.println("Relation M3: ");
		        System.out.print(m3+"\n");
		
		        Evaluation e_m3 = new Evaluation(trainingData); // set the training data on evaluation model
		        e_m3.evaluateModel(m3, testData); // evaluate the model on the test set
		        
		        // Print regression metrics
		        System.out.println("\n"+"Regression Metrics for Model (M3):"+"\n");
		        System.out.println("Mean Absolute Error (MAE): " + e_m3.meanAbsoluteError());
		        System.out.println("Root Mean Squared Error (RMSE): " + e_m3.rootMeanSquaredError());
		        System.out.println("Relative Absolute Error (RAE): " + e_m3.relativeAbsoluteError());
		        System.out.println("Correlation coefficient: " + e_m3.correlationCoefficient());
		        System.out.println("\n"+"-----------------------------------------------");
		        
		 //---------------------------------------------------------------------------  
	     //7)
		    System.out.println("7)"+"\n");
	        LinearRegression m4 = new LinearRegression();
	        m4.buildClassifier(trainingData);  // build the relation
	        System.out.println("Relation M4: "); 
	        System.out.print(m4+"\n");
	        
	        // Evaluate the model on the test set
	        Evaluation e_m4 = new Evaluation(trainingData); // set the training data on evaluation model
	        e_m4.evaluateModel(m4, testData); // evaluate the model on the test set
	
	        // Print regression metrics for Model M4
	        System.out.println("\nRegression Metrics for Model (M4):\n");
	        System.out.println("Mean Absolute Error (MAE): " + e_m4.meanAbsoluteError());
	        System.out.println("Root Mean Squared Error (RMSE): " + e_m4.rootMeanSquaredError());
	        System.out.println("Relative Absolute Error (RAE): " + e_m4.relativeAbsoluteError());
	        System.out.println("Correlation coefficient: " + e_m4.correlationCoefficient());
	        System.out.println("\n-----------------------------------------------");

	      //---------------------------------------------------------------------------  
		  //8)
	        
	      System.out.println("8)"+"\n");  
	      System.out.println("compare the performance of the generated models:"+"\n");  
	      System.out.println("- Mean Absolute Error(MAE):"+"\n");
	      System.out.println("M1: "+e_m1.meanAbsoluteError());
	      System.out.println("M2: "+e_m2.meanAbsoluteError());
	      System.out.println("M3: "+e_m3.meanAbsoluteError());
	      System.out.println("M4: "+e_m4.meanAbsoluteError());
	      
	      System.out.println("\n"+"Model M4 has the lowest Mean Absolute Error, which M4 accuracy best than the other."+"\n");
	      
	      System.out.println("---------"+"\n");
	      System.out.println("- Root Mean Squared Error(RMSE):"+"\n");
	      System.out.println("M1: "+e_m1.rootMeanSquaredError());
	      System.out.println("M2: "+e_m2.rootMeanSquaredError());
	      System.out.println("M3: "+e_m3.rootMeanSquaredError());
	      System.out.println("M4: "+e_m4.rootMeanSquaredError());
	      
	      System.out.println("\n"+"Model M4 has the lowest Root Mean Squared Error, which M4 better precision in predictions than the other."+"\n");
	      
	      System.out.println("---------"+"\n");
	      System.out.println("- Relative Absolute Error (RAE):"+"\n");
	      System.out.println("M1: "+e_m1.relativeAbsoluteError());
	      System.out.println("M2: "+e_m2.relativeAbsoluteError());
	      System.out.println("M3: "+e_m3.relativeAbsoluteError());
	      System.out.println("M4: "+e_m4.relativeAbsoluteError());
	      
	      System.out.println("\n"+"Model M4 has the lowest Relative Absolute Error, which M4 better performance in terms of relative accuracy than the other."+"\n");
	      
	      System.out.println("---------"+"\n");
	      System.out.println("- Correlation coefficient:"+"\n");
	      System.out.println("M1: "+e_m1.correlationCoefficient());
	      System.out.println("M2: "+e_m2.correlationCoefficient());
	      System.out.println("M3: "+e_m3.correlationCoefficient());
	      System.out.println("M4: "+e_m4.correlationCoefficient());
	      
	      System.out.println("\n"+"All models have the same correlation coefficient, which there is a strong linear relationship between the predicted and actual values.");
    }
    

        
    public static double calculateMedian(double[] values) {
        int middle = values.length / 2;
        if (values.length % 2 == 0) {
            return (values[middle - 1] + values[middle]) / 2.0;
        } else {
            return values[middle];
        }
    }
    	
    	
    	
    	
    	
    	
    	
   
    
}

