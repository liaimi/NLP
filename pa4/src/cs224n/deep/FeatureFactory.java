package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		//TODO implement this
		//set allVecs from filename	
		if (allVecs!=null) return allVecs;
		BufferedReader br = new BufferedReader(new FileReader(vecFilename));
		String newLine;
		double[][] wordVecs = new double[50][wordToNum.size()];
		int lineCnt = 0;
		while ((newLine = br.readLine()) != null) {
			StringTokenizer str = new StringTokenizer(newLine);
			int cnt = 0;
			while (str.hasMoreTokens()) {
				wordVecs[cnt][lineCnt] = Double.parseDouble(str.nextToken());
				cnt++;
			}
			lineCnt++;
		}
		br.close();
		allVecs = new SimpleMatrix(wordVecs);
		return allVecs;	
	}
	
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
		//TODO: create this
		if (wordToNum.size() != 0) return wordToNum;
		BufferedReader br = new BufferedReader(new FileReader(vocabFilename));
		String newLine;
		int lineCnt = 0;
		while ((newLine = br.readLine()) != null) {
			StringTokenizer str = new StringTokenizer(newLine);
			String word = str.nextToken().toString();
			wordToNum.put(word, lineCnt);
			numToWord.put(lineCnt, word);
			lineCnt++;
		}
		br.close();
		return wordToNum;
	}
	
 }
