// yousef sharbi 1202057
// anas karakra 1200467

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;

public class CovertCSVtoARFF {
	public static void main(String[] args){
		System.setProperty("com.github.fommil.netlib.BLAS", "com.github.fommil.netlib.F2jBLAS");
    	System.setProperty("com.github.fommil.netlib.LAPACK", "com.github.fommil.netlib.F2jLAPACK");
    	System.setProperty("com.github.fommil.netlib.ARPACK", "com.github.fommil.netlib.F2jARPACK");
    	
		try {
			CSVLoader loader=new CSVLoader(); 
			loader.setSource(new File("Height_Weight.csv")); // load csv file
			Instances data=loader.getDataSet(); // get all dataset from csv file
			

		    ArffSaver saver = new ArffSaver();
		    saver.setInstances(data); // set the data inside arff file
		    saver.setFile(new File("Height_Weight.arff"));
		    saver.writeBatch();
		    
		    System.out.println("the file converted from csv to arff successfully");
		}
		catch(Exception e) {
			System.out.println("error in file !!!!!");
		}
		
	}
}
