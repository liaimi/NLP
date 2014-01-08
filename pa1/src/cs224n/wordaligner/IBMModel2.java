package cs224n.wordaligner;

import cs224n.util.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple word alignment baseline model that maps source positions to target
 * positions along the diagonal of the alignment grid.
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 */
public class IBMModel2 implements WordAligner {

	private static final long serialVersionUID = 1315751943476440515L;

	private int MAX_ITER = 20;
	private double CUTOFF = 0.000001;

	// TODO: Use arrays or Counters for collecting sufficient statistics
	// from the training data.
	private CounterMap<String, String> condProb;
	private Map<String, CounterMap<Integer, Integer>> posMap;

	public Alignment align(SentencePair sentencePair) {
		// Placeholder code below.
		// TODO Implement an inference algorithm for Eq.1 in the assignment
		// handout to predict alignments based on the counts you collected with
		// train().
		Alignment alignment = new Alignment();

		List<String> targetWords = sentencePair.getTargetWords();
		List<String> sourceWords = sentencePair.getSourceWords();
		String posMapKey = targetWords.size()+" "+sourceWords.size();
		CounterMap<Integer, Integer> tmpPosMap = posMap.get(posMapKey);

		for (int srcIndex = 0; srcIndex < sourceWords.size(); srcIndex++) {
			String source = sourceWords.get(srcIndex);
			int maxIndex = -1;
			double maxValue = -1;
			for (int i = 0; i < targetWords.size(); i++) {
				String target = targetWords.get(i);
				double value = condProb.getCount(source, target)*tmpPosMap.getCount(i, srcIndex+1);
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
		IBMModel1 model1 = new IBMModel1();
		model1.train(trainingPairs);
		condProb = model1.getCondProb();
		posMap = new HashMap<String, CounterMap<Integer, Integer>>();
		int iter = 0;
		long old_time = System.currentTimeMillis();
		while (true) {
			CounterMap<String, String> counter = new CounterMap<String, String>();
			Map<String, CounterMap<Integer, Integer>> CounterPosMap = new HashMap<String, CounterMap<Integer, Integer>> ();
			// e step
			if (iter == MAX_ITER) break;
			for (SentencePair pair : trainingPairs) {
				List<String> targetWords = pair.getTargetWords();
				List<String> sourceWords = pair.getSourceWords();
				String posMapKey = targetWords.size()+" "+sourceWords.size();
				if (!posMap.containsKey(posMapKey)) {
					posMap.put(posMapKey, new CounterMap<Integer, Integer> ());
				} 
				CounterMap<Integer, Integer> tmpPosMap = posMap.get(posMapKey);
				if (!CounterPosMap.containsKey(posMapKey)) {
					CounterPosMap.put(posMapKey, new CounterMap<Integer, Integer> ());
				} 
				CounterMap<Integer, Integer> tmpCounterPosMap = CounterPosMap.get(posMapKey);
				for (int i = 0; i < targetWords.size(); i++) {
					String target = targetWords.get(i);
					double down = 0;
					if (iter == 0) {
						for (int j = 0; j < sourceWords.size(); j++) {
							String source = sourceWords.get(j);
							down += condProb.getCount(source, target)*1;
						}
						// count for null
						down += 1*condProb.getCount("null", target);

						if (down > 0) {
							for (int j = 0; j < sourceWords.size(); j++) {
								String source = sourceWords.get(j);
								double p = 1*condProb.getCount(source, target)/down;
								if (p != 0 ) {
									double tmpCount = counter.getCount(source, target);
									counter.setCount(source, target, tmpCount + p);
									double tmpPosCount = tmpCounterPosMap.getCount(i,j+1);
									tmpCounterPosMap.setCount(i, j+1, tmpPosCount + p);
								}
							}
							double p = 1* condProb.getCount("null", target)/ down;
							if (p != 0) {
								counter.setCount("null", target, counter.getCount("null", target) + p);
								tmpCounterPosMap.setCount(i,0,tmpCounterPosMap.getCount(i,0)+p);
							}
						}
					} else {
						for (int j = 0; j < sourceWords.size(); j++) {
							String source = sourceWords.get(j);
							down += condProb.getCount(source, target)*tmpPosMap.getCount(i, j+1);
						}
						down += condProb.getCount("null", target)*tmpPosMap.getCount(i, 0);
						if (down > 0) {
							for (int j = 0; j < sourceWords.size(); j++) {
								String source = sourceWords.get(j);
								double p = condProb.getCount(source, target)*tmpPosMap.getCount(i, j+1) / down;
								if (p != 0) {
									double tmpCount = counter.getCount(source, target);
									counter.setCount(source, target, tmpCount + p);
									double tmpPosCount = tmpCounterPosMap.getCount(i,j+1);
									tmpCounterPosMap.setCount(i, j+1, tmpPosCount + p);
								}
							}
							double p = tmpPosMap.getCount(i, 0)*condProb.getCount("null", target) / down;
							if (p != 0) {
								counter.setCount("null", target, counter.getCount("null", target) + p);
								tmpCounterPosMap.setCount(i,0,tmpCounterPosMap.getCount(i,0) + p);
							}
						}
					}
				}
			}
			// m step
			CounterMap<String, String> tmpcondProb = new CounterMap<String, String>();
			for (String source : counter.keySet()) {
				double down = 0;
				for (String target : counter.getCounter(source).keySet()) {
					down += counter.getCount(source, target);
				}
				if (down > 0) {
					for (String target : counter.getCounter(source).keySet()) {
						double tmp_p = counter.getCount(source, target) / down;
						if(tmp_p > CUTOFF || iter < 1){
							tmpcondProb.setCount(source, target,
									counter.getCount(source, target) / down);
						}
					}
				}
			}
			condProb = tmpcondProb;
			for (String posMapKey : CounterPosMap.keySet()) {
				CounterMap<Integer, Integer> tmpPosMap = CounterPosMap.get(posMapKey);
				for (int i : tmpPosMap.keySet()) {
					double down = 0;
					for (int j : tmpPosMap.getCounter(i).keySet()) {
						down += tmpPosMap.getCount(i, j);
					}
					if (down > 0) {

						for (int j : tmpPosMap.getCounter(i).keySet()) {
							posMap.get(posMapKey).setCount(i, j, tmpPosMap.getCount(i, j)/down);
						}
					}
				}
			}
			iter++;
			System.out.println("Iter numver:" + iter);
		}
		long new_time = System.currentTimeMillis();
		System.out.println("Time taken:"+ (new_time - old_time));
		System.out.println("Iteration number:" + iter);
	}
}