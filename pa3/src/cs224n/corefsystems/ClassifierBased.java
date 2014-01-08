package cs224n.corefsystems;

import cs224n.coref.*;
import cs224n.util.Pair;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;
import edu.stanford.nlp.util.logging.StanfordRedwoodConfiguration;

import java.text.DecimalFormat;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * @author Gabor Angeli (angeli at cs.stanford)
 */
public class ClassifierBased implements CoreferenceSystem {

	private static <E> Set<E> mkSet(E[] array){
		Set<E> rtn = new HashSet<E>();
		Collections.addAll(rtn, array);
		return rtn;
	}

	private static final Set<Object> ACTIVE_FEATURES = mkSet(new Object[]{

			/*
			 * TODO: Create a set of active features
			 */

			Feature.ExactMatch.class,
			Feature.HeadMatch.class,
			Feature.NumberMatch.class,
			Feature.GenderMatch.class,
			Feature.PronounMatch.class,
			Feature.HobbsMatch.class,
			//Feature.NERMatch.class,
			//Feature.SentenceDist.class,
			//Feature.MentionDist.class,
			//Feature.NERType.class,
			//Feature.IsCandidatePronoun.class,
			//Feature.IsFixedPronoun.class,
			//Feature.isAppositive.class,
			//Feature.isPredicateNominative.class,
			//Feature.PronounNoun.class,
			//Feature.CheckPronounType.class,

			//skeleton for how to create a pair feature
			Pair.make(Feature.ExactMatch.class, Feature.IsFixedPronoun.class),
			Pair.make(Feature.HeadWord.class, Feature.SentenceDist.class),
			Pair.make(Feature.CheckPronounType.class, Feature.MentionDist.class),
	});


	private LinearClassifier<Boolean,Feature> classifier;

	public ClassifierBased() {
		StanfordRedwoodConfiguration.setup();
		RedwoodConfiguration.current().collapseApproximate().apply();
	}

	private boolean isPredicateNominative(Mention a, Mention b) {
		if (Pronoun.isSomePronoun(a.gloss())
				|| Pronoun.isSomePronoun(b.gloss())) {
			return false;
		}
		if (a.endIndexExclusive == b.endIndexExclusive) {
			return false;
		}
		if (a.parse.getLabel().equals("NP") && b.parse.getLabel().equals("NP")) {
			if (a.sentence.equals(b.sentence)) {
				int index_end = (a.endIndexExclusive < b.endIndexExclusive) ? a.endIndexExclusive
						: b.endIndexExclusive;
				int index_begin = (a.beginIndexInclusive > b.beginIndexInclusive) ? a.beginIndexInclusive
						: b.beginIndexInclusive;

				if (Math.abs(index_begin - index_end) == 1
						&& (a.sentence.words.get(index_end).equals("is") || a.sentence.words
								.get(index_end).equals("are"))) {
					return true;
				}
			}
		}
		return false;
	}

	private boolean isAppositive(Mention a, Mention b) {
		if (Pronoun.isSomePronoun(a.gloss())
				|| Pronoun.isSomePronoun(b.gloss())) {
			return false;
		}
		if (a.parse.getLabel().equals("NP") && b.parse.getLabel().equals("NP")) {
			if (a.sentence.equals(b.sentence)) {
				int index_a = a.endIndexExclusive;
				int index_b = b.beginIndexInclusive;
				if (Math.abs(index_a - index_b) == 0) {
					Iterable<Pair<String, Integer>> output = a.sentence.parse
							.getTraversalBetween(a.beginIndexInclusive,
									b.beginIndexInclusive);
					ArrayList<Pair<String, Integer>> path = new ArrayList<Pair<String, Integer>>();
					Iterator<Pair<String, Integer>> iter = output.iterator();

					while (iter.hasNext()) {
						path.add(iter.next());
					}

					for (int i = 1; i < path.size() - 1; i++) {
						if (path.get(i).getFirst().matches("NP")
								&& path.get(i).getSecond() == 0
								&& path.get(i - 1).getFirst().matches("NP")
								&& path.get(i + 1).getFirst().matches("NP")
								&& b.headToken().nerTag().matches("PERSON")) {
							return true;
						}
					}
					return false;
				}
			}
		}
		return false;
	}

	private int hobbs(Mention mention1, Mention mention2) {
		Pronoun p = Pronoun.valueOrNull(mention1.gloss());
		if (p == null || p.speaker != Pronoun.Speaker.THIRD_PERSON) {
			return 0;
		}
		if(mention1.doc.indexOfSentence(mention1.sentence) == mention2.doc.indexOfSentence(mention2.sentence)){
			boolean top = HobbsHelper.top(mention1);
			if (!top) {
				int sen_index = mention1.doc.indexOfSentence(mention1.sentence);
				List<Mention> list = mention1.doc.getMentions();
				for (int j = mention1.doc.indexOfMention(mention1) - 1; j >= 0; j--) {
					Mention curMention = list.get(j);
					if (mention1.doc.indexOfSentence(curMention.sentence) == sen_index) {
						boolean result = HobbsHelper.getSameSentence(mention1,
								curMention);
						if (Pronoun.isSomePronoun(curMention.gloss())) {
							continue;
						}
						if (result&&curMention.equals(mention2)) {
							return 1;
						} else {
							return -1;
						}
					}
				}
			}
			return -1;
		} else {
			boolean top = HobbsHelper.top(mention1);
			if (top) {
				// find the first mention in previous sentence
				int sen_index = mention1.doc.indexOfSentence(mention1.sentence);
				for (int i = sen_index - 1; i >= 0; i--) {
					List<Mention> list = mention1.doc.getMentions();
					for (int j = 0; j < mention1.doc.indexOfMention(mention1); j++) {
						Mention curMention = list.get(j);
						if (mention1.doc.indexOfSentence(curMention.sentence) == i) {
							if (Pronoun.isSomePronoun(curMention.gloss())) {
								return -1;
							}
							if(curMention.equals(mention2)){
								return 1;
							} else {
								return -1;
							}

						}
					}
				}
			}
			return -1;
		}
	}

	public FeatureExtractor<Pair<Mention,ClusteredMention>,Feature,Boolean> extractor = new FeatureExtractor<Pair<Mention, ClusteredMention>, Feature, Boolean>() {
		private <E> Feature feature(Class<E> clazz, Pair<Mention,ClusteredMention> input, Option<Double> count){

			//--Variables
			Mention onPrix = input.getFirst(); //the first mention (referred to as m_i in the handout)
			Mention candidate = input.getSecond().mention; //the second mention (referred to as m_j in the handout)
			Entity candidateCluster = input.getSecond().entity; //the cluster containing the second mention


			//--Features
			if(clazz.equals(Feature.ExactMatch.class)){
				//(exact string match)
				return new Feature.ExactMatch(onPrix.gloss().equals(candidate.gloss()));
			} else if(clazz.equals(Feature.HeadMatch.class)) {
				/*
				 * TODO: Add features to return for specific classes. Implement calculating values of features here.
				 */
				return new Feature.HeadMatch(onPrix.headWord().toLowerCase().equals(candidate.headWord().toLowerCase()));
			} else if (clazz.equals(Feature.SentenceDist.class)) {
				int onPrixDist = onPrix.doc.indexOfSentence(onPrix.sentence);
				int candidateDist = candidate.doc.indexOfSentence(candidate.sentence);
				return new Feature.SentenceDist(onPrixDist - candidateDist);
			} else if (clazz.equals(Feature.MentionDist.class)) {
				int onPrixDist = onPrix.doc.indexOfMention(onPrix);
				int candidateDist = candidate.doc.indexOfMention(candidate);
				return new Feature.SentenceDist(onPrixDist - candidateDist);
			} else if (clazz.equals(Feature.NERType.class)) {
				return new Feature.NERType(candidate.headToken().nerTag());
			} else if (clazz.equals(Feature.IsCandidatePronoun.class)) {
				return new Feature.IsCandidatePronoun(Pronoun.isSomePronoun(candidate.gloss()));
			} else if (clazz.equals(Feature.IsFixedPronoun.class)) {
				return new Feature.IsFixedPronoun(Pronoun.isSomePronoun(onPrix.gloss()));
			} else if (clazz.equals(Feature.PronounMatch.class)) {
				int value;
				Pronoun onPrix_p = Pronoun.valueOrNull(onPrix.headWord());
				Pronoun candidate_p = Pronoun.valueOrNull(candidate.headWord());
				if (onPrix_p == null || candidate_p == null) {
					value = 0;
				} else if (onPrix_p.speaker == candidate_p.speaker
						&& onPrix_p.plural == candidate_p.plural && onPrix_p.gender == candidate_p.gender) {
					value = 1;
				} else {
					value = -1;
				}
				return new Feature.PronounMatch(value);				
			} else if(clazz.equals(Feature.PronounNoun.class)) {
				return new Feature.PronounNoun(!Pronoun.isSomePronoun(onPrix.gloss()) && Pronoun.isSomePronoun(candidate.gloss()));
			}
			else if(clazz.equals(Feature.HobbsMatch.class)) {
				return new Feature.HobbsMatch(hobbs(onPrix, candidate));
			}
			else if(clazz.equals(Feature.NERMatch.class)) {
				String onPrixNER = onPrix.headToken().nerTag();
				String candidateNER = candidate.headToken().nerTag();
				return new Feature.NERMatch(onPrixNER.equals(candidateNER));
			} else if(clazz.equals(Feature.isAppositive.class)) {
				return new Feature.isAppositive(isAppositive(onPrix, candidate));
			} else if(clazz.equals(Feature.isPredicateNominative.class)) {
				return new Feature.isPredicateNominative(isPredicateNominative(onPrix, candidate));
			} else if(clazz.equals(Feature.HeadWord.class)) {
				return new Feature.HeadWord(candidate.headWord());
			} else if (clazz.equals(Feature.CheckPronounType.class)) {
				Pronoun onPrix_p = Pronoun.valueOrNull(onPrix.headWord());
				if (onPrix_p != null && 
						(onPrix_p.type == Pronoun.Type.POSESSIVE_DETERMINER || onPrix_p.type == Pronoun.Type.POSESSIVE_PRONOUN)) {
					return new Feature.CheckPronounType(true);
				} else {
					return new Feature.CheckPronounType(false);
				} 
			} else if (clazz.equals(Feature.NumberMatch.class)) {
				int value;
				Pronoun onPrix_p = Pronoun.valueOrNull(onPrix.headWord());
				Pronoun candidate_p = Pronoun.valueOrNull(candidate.headWord());
				Pair<Boolean, Boolean> sameNumber = Util.haveNumberAndAreSameNumber(onPrix, candidate);
				if (onPrix_p != null && candidate_p != null) {
					value = 0;
				} else if (sameNumber.getFirst() && sameNumber.getSecond()) {
					value = 1;
				} else {
					value = 0;
				}
				return new Feature.NumberMatch(value);
			} else if (clazz.equals(Feature.GenderMatch.class)) {
				int value;
				Pronoun onPrix_p = Pronoun.valueOrNull(onPrix.headWord());
				Pronoun candidate_p = Pronoun.valueOrNull(candidate.headWord());
				Pair<Boolean, Boolean> sameGender = Util.haveGenderAndAreSameGender(onPrix, candidate);
				if (onPrix_p != null && candidate_p != null) {
					value = 0;
				} else if (sameGender.getFirst() && sameGender.getSecond()) {
					value = 1;
				} else {
					value = 0;
				}
				return new Feature.GenderMatch(value);
			} else {
				throw new IllegalArgumentException("Unregistered feature: " + clazz);
			}
		}

		@SuppressWarnings({"unchecked"})
		@Override
		protected void fillFeatures(Pair<Mention, ClusteredMention> input, Counter<Feature> inFeatures, Boolean output, Counter<Feature> outFeatures) {
			//--Input Features
			for(Object o : ACTIVE_FEATURES){
				if(o instanceof Class){
					//(case: singleton feature)
					Option<Double> count = new Option<Double>(1.0);
					Feature feat = feature((Class) o, input, count);
					if(count.get() > 0.0){
						inFeatures.incrementCount(feat, count.get());
					}
				} else if(o instanceof Pair){
					//(case: pair of features)
					Pair<Class,Class> pair = (Pair<Class,Class>) o;
					Option<Double> countA = new Option<Double>(1.0);
					Option<Double> countB = new Option<Double>(1.0);
					Feature featA = feature(pair.getFirst(), input, countA);
					Feature featB = feature(pair.getSecond(), input, countB);
					if(countA.get() * countB.get() > 0.0){
						inFeatures.incrementCount(new Feature.PairFeature(featA, featB), countA.get() * countB.get());
					}
				}
			}

			//--Output Features
			if(output != null){
				outFeatures.incrementCount(new Feature.CoreferentIndicator(output), 1.0);
			}
		}

		@Override
		protected Feature concat(Feature a, Feature b) {
			return new Feature.PairFeature(a,b);
		}
	};

	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		startTrack("Training");
		//--Variables
		RVFDataset<Boolean, Feature> dataset = new RVFDataset<Boolean, Feature>();
		LinearClassifierFactory<Boolean, Feature> fact = new LinearClassifierFactory<Boolean,Feature>();
		//--Feature Extraction
		startTrack("Feature Extraction");
		for(Pair<Document,List<Entity>> datum : trainingData) {
			//(document variables)
			Document doc = datum.getFirst();
			List<Entity> goldClusters = datum.getSecond();
			List<Mention> mentions = doc.getMentions();
			Map<Mention,Entity> goldEntities = Entity.mentionToEntityMap(goldClusters);
			startTrack("Document " + doc.id);
			//(for each mention...)
			for(int i=0; i<mentions.size(); i++){
				//(get the mention and its cluster)
				Mention onPrix = mentions.get(i);
				Entity source = goldEntities.get(onPrix);
				if(source == null){ throw new IllegalArgumentException("Mention has no gold entity: " + onPrix); }
				//(for each previous mention...)
				int oldSize = dataset.size();
				for(int j=i-1; j>=0; j--){
					//(get previous mention and its cluster)
					Mention cand = mentions.get(j);
					Entity target = goldEntities.get(cand);
					if(target == null){ throw new IllegalArgumentException("Mention has no gold entity: " + cand); }
					//(extract features)
					Counter<Feature> feats = extractor.extractFeatures(Pair.make(onPrix, cand.markCoreferent(target)));
					//(add datum)
					dataset.add(new RVFDatum<Boolean, Feature>(feats, target == source));
					//(stop if
					if(target == source){ break; }
				}
				//logf("Mention %s (%d datums)", onPrix.toString(), dataset.size() - oldSize);
			}
			endTrack("Document " + doc.id);
		}
		endTrack("Feature Extraction");
		//--Train Classifier
		startTrack("Minimizer");
		this.classifier = fact.trainClassifier(dataset);
		endTrack("Minimizer");
		//--Dump Weights
		startTrack("Features");
		//(get labels to print)
		Set<Boolean> labels = new HashSet<Boolean>();
		labels.add(true);
		//(print features)
		for(Triple<Feature,Boolean,Double> featureInfo : this.classifier.getTopFeatures(labels, 0.0, true, 100, true)){
			Feature feature = featureInfo.first();
			Boolean label = featureInfo.second();
			Double magnitude = featureInfo.third();
			//log(FORCE,new DecimalFormat("0.000").format(magnitude) + " [" + label + "] " + feature);
		}
		end_Track("Features");
		endTrack("Training");
	}

	public List<ClusteredMention> runCoreference(Document doc) {
		//--Overhead
		startTrack("Testing " + doc.id);
		//(variables)
		List<ClusteredMention> rtn = new ArrayList<ClusteredMention>(doc.getMentions().size());
		List<Mention> mentions = doc.getMentions();
		int singletons = 0;
		//--Run Classifier
		for(int i=0; i<mentions.size(); i++){
			//(variables)
			Mention onPrix = mentions.get(i);
			int coreferentWith = -1;
			//(get mention it is coreferent with)
			for(int j=i-1; j>=0; j--){
				ClusteredMention cand = rtn.get(j);
				boolean coreferent = classifier.classOf(new RVFDatum<Boolean, Feature>(extractor.extractFeatures(Pair.make(onPrix, cand))));
				if(coreferent){
					coreferentWith = j;
					break;
				}
			}
			//(mark coreference)
			if(coreferentWith < 0){
				singletons += 1;
				rtn.add(onPrix.markSingleton());
			} else {
				//log("Mention " + onPrix + " coreferent with " + mentions.get(coreferentWith));
				rtn.add(onPrix.markCoreferent(rtn.get(coreferentWith)));
			}
		}
		//log("" + singletons + " singletons");
		//--Return
		endTrack("Testing " + doc.id);
		//System.out.println(rtn);
		//System.out.println(doc.sentences);
		//System.out.println();
		return rtn;
	}

	private class Option<T> {
		private T obj;
		public Option(T obj){ this.obj = obj; }
		public Option(){};
		public T get(){ return obj; }
		public void set(T obj){ this.obj = obj; }
		public boolean exists(){ return obj != null; }
	}
}
