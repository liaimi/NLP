package cs224n.wordaligner;

import cs224n.util.*;

import java.util.List;

/**
 * Simple word alignment baseline model that maps source positions to target
 * positions along the diagonal of the alignment grid.
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 */
public class IBMModel1 implements WordAligner {

	private static final long serialVersionUID = 1315751943476440515L;

	private CounterMap<String, String> condProb;
	private int MAX_ITER = 20;
	private double CUTOFF = 0.000001;

	public CounterMap<String, String> getCondProb() {
		return condProb;
	}

	public Alignment align(SentencePair sentencePair) {
		// Placeholder code below.
		// TODO Implement an inference algorithm for Eq.1 in the assignment
		// handout to predict alignments based on the counts you collected with
		// train().
		Alignment alignment = new Alignment();

		List<String> targetWords = sentencePair.getTargetWords();
		List<String> sourceWords = sentencePair.getSourceWords();
		for (int srcIndex = 0; srcIndex < sourceWords.size(); srcIndex++) {
			String source = sourceWords.get(srcIndex);
			int maxIndex = -1;
			double maxValue = -1;
			for (int i = 0; i < targetWords.size(); i++) {
				String target = targetWords.get(i);
				double value = condProb.getCount(source, target);
				if (maxValue < value) {
					maxValue = value;
					maxIndex = i;
				}
			}
			if (maxIndex != -1) {
				alignment.addPredictedAlignment(maxIndex , srcIndex);
			}
		}
		return alignment;
	}

	public void train(List<SentencePair> trainingPairs) {
		condProb = new CounterMap<String, String>();
		int iter = 0;
		long old_time = System.currentTimeMillis();
		while (true) {
			if (iter == MAX_ITER) break;
			CounterMap<String, String> counter = new CounterMap<String, String>();
			// e step
			for (SentencePair pair : trainingPairs) {
				List<String> targetWords = pair.getTargetWords();
				List<String> sourceWords = pair.getSourceWords();
				for (String target : targetWords) {
					double down = 0;
					if (iter == 0) {
						// use 1 for initialized value
						down += targetWords.size();
						down += 1;
						for (String source : sourceWords) {
							double p = 1 / down;
							double tmpCount = counter.getCount(source, target);
							counter.setCount(source, target, tmpCount + p);
						}
						counter.setCount("null", target, counter.getCount("null", target) + 1 / down);
					} else {
						for (String source : sourceWords) {
							down += condProb.getCount(source, target);
						}
						down += condProb.getCount("null", target);
						for (String source : sourceWords) {
							double p = condProb.getCount(source, target) / down;
							if(p!=0){
								double tmpCount = counter.getCount(source, target);
								counter.setCount(source, target, tmpCount + p);
							}
						}
						counter.setCount("null", target, counter.getCount("null", target) + 
								condProb.getCount("null", target) / down);
					}
				}
			}

			CounterMap<String, String> tmpcondProb = new CounterMap<String, String>();
			// m step
			for (String source : counter.keySet()) {
				double down = 0;
				for (String target : counter.getCounter(source).keySet()) {
					down += counter.getCount(source, target);
				}
				for (String target : counter.getCounter(source).keySet()) {
					double tmp_p = counter.getCount(source, target) / down;
					if(tmp_p > CUTOFF || iter < 1){
						tmpcondProb.setCount(source, target, tmp_p);
					}
				}
			}
			condProb = tmpcondProb;
			iter++;
			System.out.println("Iter number:" + iter);
		}
		long new_time = System.currentTimeMillis();
		System.out.println("Time taken:"+ (new_time - old_time));
		System.out.println("Iteration number:" + iter);
	}
}