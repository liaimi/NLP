package cs224n.deep;

import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel {
    
	public final String persontrainLabel = "PERSON";
	protected SimpleMatrix L, W, Wout, H;
	public int windowSize, wordSize, hiddenSize;
	public double lr;
	double C = 0.0001;
	public final double epsilon = 10e-4;
	boolean regularized;
	int trainingSize;
	int iters;
    
	public WindowModel(double C, int iters, double _lr, int _windowSize,
                       int _hiddenSize) {
		hiddenSize = _hiddenSize;
		windowSize = _windowSize;
		wordSize = FeatureFactory.allVecs.numRows();
		lr = _lr;
		regularized = true;
		this.C = C;
		this.iters = iters;
	}
    
	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		// TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix... hidden size X ( window * 50 + 1)
		// U for the score (hidden size + 1) X 1
		// U = SimpleMatrix...
		double eta = Math.sqrt(6) / (Math.sqrt(windowSize * wordSize + hiddenSize));
		L = FeatureFactory.allVecs;
        
		// SimpleMatrix matrix = L.extractVector(true, 0);
		// SimpleMatrix matrix1 = L.extractVector(true, 1);
		// SimpleMatrix matrix2 = L.extractVector(true, L.numRows());
		Random random = new Random();
		W = SimpleMatrix.random(hiddenSize, windowSize * wordSize + 1,
                                -1 * eta, eta, random);
		Wout = SimpleMatrix.random(hiddenSize + 1, 1, -1 * eta, eta, random);
	}
    
	public double getG(double input) {
		return 1.0 / (1.0 + Math.exp(-1 * input));
	}
    
	public SimpleMatrix getF(SimpleMatrix input) {
		for (int i = 0; i < input.numRows(); i++) {
			input.set(i, 0, Math.tanh(input.get(i, 0)));
		}
		return input;
	}
    
	public SimpleMatrix getFPrime(SimpleMatrix input) {
		for (int i = 0; i < input.numRows(); i++) {
			input.set(i, 0,
                      1 - Math.tanh(input.get(i, 0)) * Math.tanh(input.get(i, 0)));
		}
		return input;
	}
    
	public double getH(SimpleMatrix x) {
		SimpleMatrix f_output = getF(W.mult(x));
		SimpleMatrix f_dot = new SimpleMatrix(hiddenSize + 1, 1);
		f_dot.insertIntoThis(0, 0, f_output);
		SimpleMatrix b2 = new SimpleMatrix(1, 1);
		b2.set(1);
		f_dot.insertIntoThis(f_dot.numRows() - 1, 0, b2);
		double g_input = Wout.dot(f_dot); 
		return getG(g_input);
	}
    
	public void updateWout(SimpleMatrix x, String label) {
		double y = 0;
		if (label.matches(persontrainLabel)) {
			y = 1;
		}
		double delta = -1 * y * (1 - getH(x)) + (1 - y) * getH(x);
		SimpleMatrix fx = getF(W.mult(x));
		SimpleMatrix u_diff = new SimpleMatrix(fx.numRows() + 1, fx.numCols());
        
		u_diff.insertIntoThis(0, 0, fx);
		SimpleMatrix b2 = new SimpleMatrix(1, 1);
		b2.set(1);
		u_diff.insertIntoThis(u_diff.numRows() - 1, 0, b2);
        
		SimpleMatrix diff = u_diff.scale(delta * lr);
        
		if (regularized) {
			SimpleMatrix regularizedTerm = Wout.copy();
			regularizedTerm = regularizedTerm.scale(C
                                                    / (trainingSize - windowSize));
			Wout = Wout.minus(regularizedTerm);
		}
		Wout = Wout.minus(diff);
	}
    
	public void updateL(SimpleMatrix x, String label,
                        LinkedList<Integer> word_indices) {
		double y = 0;
		if (label.matches(persontrainLabel)) {
			y = 1;
		}
		double delta = -1 * y * (1 - getH(x)) + (1 - y) * getH(x);
        
		SimpleMatrix f_output = getFPrime(W.mult(x));
		for (int i = 0; i < f_output.numRows(); i++) {
			f_output.set(i, 0, f_output.get(i, 0) * Wout.get(i, 0));
		}
        
		SimpleMatrix fx = W.transpose().mult(f_output);
		SimpleMatrix diff = fx.scale(delta * lr);
		for (int c = 0; c < windowSize; c++) {
			for (int i = 0; i < wordSize; i++) {
				L.set(i, word_indices.get(c), L.get(i, word_indices.get(c))
                      - diff.get(c * wordSize + i, 0));
			}
		}
	}
    
	public void updateW(SimpleMatrix x, String label) {
		double y = 0;
		if (label.matches(persontrainLabel)) {
			y = 1;
		}
		double delta = -1 * y * (1 - getH(x)) + (1 - y) * getH(x);
        
		SimpleMatrix f_output = getFPrime(W.mult(x));
		for (int i = 0; i < f_output.numRows(); i++) {
			f_output.set(i, 0, f_output.get(i, 0) * Wout.get(i, 0));
		}
		SimpleMatrix fx = f_output.mult(x.transpose());
		SimpleMatrix diff = fx.scale(delta * lr);
        
		if (regularized) {
			SimpleMatrix regularizedTerm = W.copy();
			regularizedTerm = regularizedTerm.scale(C / (trainingSize - windowSize));
			W = W.minus(regularizedTerm);
		}
        
		W = W.minus(diff);
	}
    
	public double getJ(List<Datum> _trainData) {
		double J = 0;
		for (int i = windowSize / 2; i < _trainData.size() - windowSize / 2; i++) {
			SimpleMatrix x = new SimpleMatrix(windowSize * wordSize + 1, 1);
			for (int k = 0; k < windowSize; k++) {
				Datum cur = _trainData.get(i - windowSize / 2 + k);
				HashMap<String, Integer> map = FeatureFactory.wordToNum;
				int word_index = -1;
				// System.out.println(cur.word);
				if (map.containsKey(cur.word.toLowerCase())) {
					word_index = map.get(cur.word.toLowerCase());
				} else {
					word_index = 0;
					// System.out.println(cur.word);
				}
				SimpleMatrix word = L.extractVector(false, word_index);
				x.insertIntoThis(k * wordSize, 0, word);
			}
			SimpleMatrix b1 = new SimpleMatrix(1, 1);
			b1.set(1);
			x.insertIntoThis(x.numRows() - 1, 0, b1);
            
			double y = 0;
			if (_trainData.get(i).label.matches(persontrainLabel)) {
				y = 1;
			}
			J += -1 * y * Math.log(getH(x)) - (1 - y) * Math.log(1 - getH(x));
		}
		J /= _trainData.size() - windowSize;
		return J;
	}
    
	public double getJPrimeForWout(List<Datum> _trainData, int index) {
		double output = 0;
		for (int i = windowSize / 2; i < _trainData.size() - windowSize / 2; i++) {
			SimpleMatrix x = new SimpleMatrix(windowSize * wordSize + 1, 1);
			for (int k = 0; k < windowSize; k++) {
				Datum cur = _trainData.get(i - windowSize / 2 + k);
				HashMap<String, Integer> map = FeatureFactory.wordToNum;
				int word_index = -1;
				// System.out.println(cur.word);
				if (map.containsKey(cur.word.toLowerCase())) {
					word_index = map.get(cur.word.toLowerCase());
				} else {
					word_index = 0;
					// System.out.println(cur.word);
				}
				SimpleMatrix word = L.extractVector(false, word_index);
				x.insertIntoThis(k * wordSize, 0, word);
			}
			SimpleMatrix b1 = new SimpleMatrix(1, 1);
			b1.set(1);
			x.insertIntoThis(x.numRows() - 1, 0, b1);
			double y = 0;
			if (_trainData.get(i).label.matches(persontrainLabel)) {
				y = 1;
			}
            
			double delta = -1 * y * (1 - getH(x)) + (1 - y) * getH(x);
			SimpleMatrix fx = getF(W.mult(x));
			SimpleMatrix u_diff = new SimpleMatrix(fx.numRows() + 1,
                                                   fx.numCols());
            
			u_diff.insertIntoThis(0, 0, fx);
			SimpleMatrix b2 = new SimpleMatrix(1, 1);
			b2.set(1);
			u_diff.insertIntoThis(u_diff.numRows() - 1, 0, b2);
            
			SimpleMatrix diff = u_diff.scale(delta);
			output += diff.get(index);
		}
		output /= _trainData.size() - windowSize;
		return output;
	}
    
	public double getJPrimeForL(List<Datum> _trainData, int row_index,
                                int col_index) {
		double output = 0;
		for (int i = windowSize / 2; i < _trainData.size() - windowSize / 2; i++) {
			SimpleMatrix x = new SimpleMatrix(windowSize * wordSize + 1, 1);
			LinkedList<Integer> indices = new LinkedList<Integer>();
			for (int k = 0; k < windowSize; k++) {
				Datum cur = _trainData.get(i - windowSize / 2 + k);
				HashMap<String, Integer> map = FeatureFactory.wordToNum;
				int word_index = -1;
				// System.out.println(cur.word);
				if (map.containsKey(cur.word.toLowerCase())) {
					word_index = map.get(cur.word.toLowerCase());
				} else {
					word_index = 0;
					// System.out.println(cur.word);
				}
				indices.add(word_index);
				SimpleMatrix word = L.extractVector(false, word_index);
				x.insertIntoThis(k * wordSize, 0, word);
			}
			SimpleMatrix b1 = new SimpleMatrix(1, 1);
			b1.set(1);
			x.insertIntoThis(x.numRows() - 1, 0, b1);
			double y = 0;
			if (_trainData.get(i).label.matches("persontrainLabel")) {
				y = 1;
			}
            
			double delta = -1 * y * (1 - getH(x)) + (1 - y) * getH(x);
            
			SimpleMatrix f_output = getFPrime(W.mult(x));
			for (int ii = 0; ii < f_output.numRows(); ii++) {
				f_output.set(ii, 0, f_output.get(ii, 0) * Wout.get(ii, 0));
			}
            
			SimpleMatrix fx = W.transpose().mult(f_output);
			SimpleMatrix diff = fx.scale(delta);
			for (int c = 0; c < windowSize; c++) {
				if (indices.get(c) == col_index) {
					// System.out.println("here");
					output += diff.get(c * wordSize + row_index, 0);
				}
			}
		}
		output /= _trainData.size() - windowSize;
		return output;
	}
    
	public double getJPrimeForW(List<Datum> _trainData, int row_index,
                                int col_index) {
		double output = 0;
		for (int i = windowSize / 2; i < _trainData.size() - windowSize / 2; i++) {
			SimpleMatrix x = new SimpleMatrix(windowSize * wordSize + 1, 1);
			for (int k = 0; k < windowSize; k++) {
				Datum cur = _trainData.get(i - windowSize / 2 + k);
				HashMap<String, Integer> map = FeatureFactory.wordToNum;
				int word_index = -1;
				// System.out.println(cur.word);
				if (map.containsKey(cur.word.toLowerCase())) {
					word_index = map.get(cur.word.toLowerCase());
				} else {
					word_index = 0;
					// System.out.println(cur.word);
				}
				SimpleMatrix word = L.extractVector(false, word_index);
				x.insertIntoThis(k * wordSize, 0, word);
			}
			SimpleMatrix b1 = new SimpleMatrix(1, 1);
			b1.set(1);
			x.insertIntoThis(x.numRows() - 1, 0, b1);
			double y = 0;
			if (_trainData.get(i).label.matches(persontrainLabel)) {
				y = 1;
			}
            
			double delta = -1 * y * (1 - getH(x)) + (1 - y) * getH(x);
			SimpleMatrix f_output = getFPrime(W.mult(x));
			for (int k = 0; k < f_output.numRows(); k++) {
				f_output.set(k, 0, f_output.get(k, 0) * Wout.get(k, 0));
			}
			SimpleMatrix fx = f_output.mult(x.transpose());
			SimpleMatrix diff = fx.scale(delta);
			output += diff.get(row_index, col_index);
		}
		output /= _trainData.size() - windowSize;
		return output;
	}
    
	public boolean gradientCheck(List<Datum> _trainData) {
		// for(int wr = 0; wr < W.numRows(); wr++){
		// for(int wc = 0; wc < W.numCols(); wc++){
		// W.set(wr, wc, W.get(wr, wc)-0.0001);
		// double J1 = getJ(_trainData);
		// W.set(wr, wc, W.get(wr, wc)+0.0002);
		// double J2 = getJ(_trainData);
		// W.set(wr, wc, W.get(wr, wc)-0.0001);
		// double f = getJPrimeForW(_trainData, wr, wc);
		// double right = (J2-J1)/2/0.0001;
		// if(Math.abs(f-right)>0.0000001){
		// System.out.println("W left:"+f);
		// System.out.println("W right:"+right);
		// System.out.println();
		// return false;
		// }
		// }
		// }
        
		for (int i = 0; i < Wout.numRows(); i++) {
			Wout.set(i, 0, Wout.get(i, 0) - 0.0001);
			double J1 = getJ(_trainData);
			Wout.set(i, 0, Wout.get(i, 0) + 0.0002);
			double J2 = getJ(_trainData);
			Wout.set(i, 0, Wout.get(i, 0) - 0.0001);
			double right = (J2 - J1) / 2 / 0.0001;
			double f = getJPrimeForWout(_trainData, i);
			System.out.println("Wout  left:" + f);
			System.out.println("Wout right:" + right);
			System.out.println();
		}
        
		System.out.println(L.numCols());
		System.out.println(L.numRows());
		// for (int lr = 0; lr < L.numRows(); lr++) {
		// for (int lc = 0; lc < L.numCols(); lc++) {
		// L.set(lr, lc, L.get(lr, lc) - 0.0001);
		// double J1 = getJ(_trainData);
		// L.set(lr, lc, L.get(lr, lc) + 0.0002);
		// double J2 = getJ(_trainData);
		// L.set(lr, lc, L.get(lr, lc) - 0.0001);
		// double right = (J2 - J1) / 2 / 0.0001;
		// double f = getJPrimeForL(_trainData, lr, lc);
		// System.out.println("L  left:" + f);
		// System.out.println("L right:" + right);
		// System.out.println();
		// }
		// }
		return true;
	}
    
	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> _trainData) {
		trainingSize = _trainData.size();
		for (int k = 0; k < iters; k++) {
			// need to handle start and end
			for (int i = windowSize / 2; i < _trainData.size() - windowSize / 2; i++) {
//				if (i % 10000 == 0) {
//					System.out.println(i);
//				}
				SimpleMatrix x = new SimpleMatrix(windowSize * wordSize + 1, 1);
				LinkedList<Integer> word_indices = new LinkedList<Integer>();
				for (int j = 0; j < windowSize; j++) {
					Datum cur = _trainData.get(i - windowSize / 2 + j);
					HashMap<String, Integer> map = FeatureFactory.wordToNum;
					int word_index = -1;
					// System.out.println(cur.word);
					if (map.containsKey(cur.word.toLowerCase())) {
						word_index = map.get(cur.word.toLowerCase());
					} else {
						word_index = 0;
						// System.out.println(cur.word);
					}
					word_indices.add(word_index);
					SimpleMatrix word = L.extractVector(false, word_index);
					x.insertIntoThis(j * wordSize, 0, word);
				}
				SimpleMatrix b1 = new SimpleMatrix(1, 1);
				b1.set(1);
				x.insertIntoThis(x.numRows() - 1, 0, b1);
                
				updateWout(x, _trainData.get(i).label);
				updateW(x, _trainData.get(i).label);
				updateL(x, _trainData.get(i).label, word_indices);
				// gradientCheck(_trainData);
				// System.out.println(L.get(1,1));
			}
		}
		//System.out.println(Wout);
	}
    
	// public double gradientCheck(double )
    
	public void test(List<Datum> testData) {
		// TODO
		H = new SimpleMatrix(testData.size(), 1);
		int offset = (windowSize - 1) / 2;
		SimpleMatrix W_extracted = W.extractMatrix(0, W.numRows(), 0,
                                                   W.numCols() - 1);
		SimpleMatrix b_1 = W.extractMatrix(0, W.numRows(), W.numCols() - 1,
                                           W.numCols());
		for (int i = offset; i < testData.size() - offset; i++) {
			SimpleMatrix x = new SimpleMatrix(windowSize * wordSize, 1);
			// System.out.println(testData.get(i).word);
			for (int j = -1 * offset; j <= offset; j++) {
				if ((i + j) >= 0 && (i + j) < testData.size()) {
					String word = testData.get(i + j).word.toLowerCase();
					// System.out.println(word);
					int wordIndex;
					if (FeatureFactory.wordToNum.containsKey(word)) {
						wordIndex = FeatureFactory.wordToNum.get(word);
					} else {
						wordIndex = 0;
					}
					SimpleMatrix toAdd = L.extractMatrix(0, L.numRows(), wordIndex, wordIndex + 1);
					x.insertIntoThis((offset + j) * wordSize, 0, toAdd);
				}
			}
            
			SimpleMatrix z = W_extracted.mult(x).plus(b_1);
			SimpleMatrix a = getF(z);
			double[][] b_2 = new double[1][1];
			b_2[0][0] = 1;
			SimpleMatrix a_new = new SimpleMatrix(a.numRows() + 1, a.numCols());
			a_new.insertIntoThis(0, 0, a);
			a_new.insertIntoThis(hiddenSize, 0, new SimpleMatrix(b_2));
			// a.combine(a.END+1,a.END+1,new SimpleMatrix(b_2));
			// SimpleMatrix Wout_extracted =
			// Wout.extractMatrix(0,hiddenSize,0,1);
			// double h =
			// getG(Wout_extracted.transpose().mult(a).get(0,0)+Wout.get(hiddenSize,0));
			double h = getG(Wout.transpose().mult(a_new).get(0, 0));
			if (i < offset || i >= testData.size() - offset || h < 0.5) {
				H.set(i, 0, 0);
			} else {
				// person entity
				H.set(i, 0, 1);
			}
		}
		//System.out.println(H.toString());
	}
    
}
