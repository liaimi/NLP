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
public class PMIModel implements WordAligner {

	private static final long serialVersionUID = 1315751943476440515L;

	// TODO: Use arrays or Counters for collecting sufficient statistics
	// from the training data.
	private CounterMap<String,String> sourceTargetCounts;
	private Counter<String> sourceCounts;
	private Counter<String> targetCounts;

	public Alignment align(SentencePair sentencePair) {
		// Placeholder code below. 
		// TODO Implement an inference algorithm for Eq.1 in the assignment
		// handout to predict alignments based on the counts you collected with train().
		Alignment alignment = new Alignment();
		int numSourceWords = sentencePair.getSourceWords().size();
		int numTargetWords = sentencePair.getTargetWords().size();
		for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
			int maxIndex = -1;
			double maxValue = 0;
			String source = sentencePair.getSourceWords().get(srcIndex);
			for (int tgtIndex = 0; tgtIndex < numTargetWords; tgtIndex++) {
				String target = sentencePair.getTargetWords().get(tgtIndex);
				double score = sourceTargetCounts.getCount(source,target);
				score = score/sourceCounts.getCount(source)/targetCounts.getCount(target);
				if(score > maxValue) {
					maxIndex = tgtIndex;
					maxValue = score;
				}
			}
			alignment.addPredictedAlignment(maxIndex, srcIndex);
		}
		return alignment;
	}

	public void train(List<SentencePair> trainingPairs) {
		sourceTargetCounts = new CounterMap<String,String>();
		sourceCounts = new Counter<String>();
		targetCounts = new Counter<String>();
		sourceCounts.setCount("null", trainingPairs.size());
		for(SentencePair pair : trainingPairs){
			List<String> targetWords = pair.getTargetWords();
			List<String> sourceWords = pair.getSourceWords();
			for(String source : sourceWords){
				double count = sourceCounts.getCount(source);
				sourceCounts.setCount(source, count+1.0);
			}
			for(String target : targetWords){
				double count = targetCounts.getCount(target);
				targetCounts.setCount(target, count+1.0);
				count = sourceTargetCounts.getCount("null", target);
				sourceTargetCounts.setCount("null", target, count+1.0);
			}
			for(String source : sourceWords){
				for(String target : targetWords){
					double count = sourceTargetCounts.getCount(source, target);
					sourceTargetCounts.setCount(source, target, count + 1.0);
				}
			}
		}
	}
}