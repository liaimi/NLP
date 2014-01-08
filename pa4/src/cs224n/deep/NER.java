package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {

	public final static String persontestLabel = "0";

	public static double getF1Score(List<Datum> testData, SimpleMatrix H) {
		int posCnt = 0;
		int trueCnt = 0;
		int correctCnt = 0;
		for (int i = 0; i < testData.size(); i++) {
			if (testData.get(i).label.matches(persontestLabel)) {
				trueCnt++;
			}
			if (H.get(i) == 1) {
				posCnt++;
			} 
			if (testData.get(i).label.matches(persontestLabel) && H.get(i) == 1) {
				correctCnt++;
			}
		}
		double precision = (double)correctCnt/posCnt;
		double recall = (double)correctCnt/trueCnt;
		System.out.println("Precision: "+ precision);
		System.out.println("Recall: " + recall);
		return 2*(precision*recall)/(precision+recall);
	}

	public static void printLMatrix(SimpleMatrix L) {
		try {
			BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter("wordVectors_after"));
			for (int i = 0; i < L.numCols(); i++) {
				for (int j = 0; j < L.numRows(); j++) {
					bufferedWriter.write(Double.toString(L.get(j, i)) + " ");
				}
				bufferedWriter.newLine();
			}
			bufferedWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void main(String[] args) throws IOException {
		
		if (args.length < 2) {
			System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
			return;
		}	    

		//String[] args2 = {"/Users/aimee/cs224n/pa4/data/train2","/Users/aimee/cs224n/pa4/data/dev2"};
		// this reads in the train and test datasets
		List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
		List<Datum> testData = FeatureFactory.readTestData(args[1]);	

		//	read the train and test data
		//TODO: Implement this function (just reads in vocab and word vectors)
		FeatureFactory.initializeVocab("../data/vocab.txt");
		SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors.txt");

		// initialize model 
		WindowModel model = new WindowModel(0.01, 10, 0.01, 5, 100);
		model.initWeights();

		//TODO: Implement those two functions
		model.train(trainData);
		//printLMatrix(model.L);
		model.test(testData);
		System.out.println("F1 score:" + getF1Score(testData, model.H));
	}
	
}