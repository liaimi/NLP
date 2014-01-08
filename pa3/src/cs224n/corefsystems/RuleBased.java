package cs224n.corefsystems;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.coref.Mention;
import cs224n.coref.Pronoun;
import cs224n.coref.Util;
import cs224n.util.Pair;

public class RuleBased implements CoreferenceSystem {

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {

	}

	private void merge(Set<Mention> m1, Set<Mention> m2) {
		m1.addAll(m2);
		m2.removeAll(m2);
	}

	private boolean isSameHead(Set<Mention> one, Set<Mention> two) {
		boolean isCoref = false;
		for (Mention a : one) {
			for (Mention b : two) {
				if (!Pronoun.isSomePronoun(a.gloss())
						&& !Pronoun.isSomePronoun(b.gloss())) {
					if (a.headWord().equalsIgnoreCase(b.headWord())) {
						isCoref = true;
						break;
					}
				}
			}
		}
		return isCoref;
	}

	private Set<String> getModifiers(Mention a) {
		Set<String> modifiers = new HashSet<String>();
		for (int i = a.beginIndexInclusive; i < a.endIndexExclusive; i++) {

			if (a.sentence.tokens.get(i).isNoun()) {
				modifiers.add(a.sentence.words.get(i));
			}
		}
		return modifiers;
	}

	private boolean sameModifiers(Set<Mention> one, Set<Mention> two) {
		boolean isCoref = false;
		for (Mention a : one) {
			if (!Pronoun.isSomePronoun(a.gloss())) {
				Set<String> modifiers_a = getModifiers(a);
				if (modifiers_a.size() < 1) {
					continue;
				}
				for (Mention b : two) {
					if (!Pronoun.isSomePronoun(b.gloss())) {

						Position a_pos = new Position(a);
						Position b_pos = new Position(b);
						if (a_pos.distance(b_pos) == -1) {
							return false;
						}

						if (modifiers_a.contains(b.gloss())) {
							isCoref = true;
							break;
						}
					}
				}
			}
		}
		return isCoref;
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
//							System.out.println(b.headToken().nerTag());
//							System.out.println(a.headToken().nerTag());
//							System.out.println(a.gloss());
//							System.out.println(b.gloss());
//							System.out.println(a.sentence);
//							System.out.println(output);
//							System.out.println();
							return true;
						}
					}

					return false;
				}
			}
		}
		return false;
	}

	private boolean isAppositive(Set<Mention> one, Set<Mention> two) {
		boolean isCoref = false;
		for (Mention a : one) {
			for (Mention b : two) {
				if (isAppositive(a, b)) {
					isCoref = true;
				}
			}
		}
		return isCoref;
	}

	private Mention hobbsPreSentence(HashSet<Mention> m1) {
		if (m1.size() == 0) {
			return null;
		}
		for (Mention mention1 : m1) {
			Pronoun p = Pronoun.valueOrNull(mention1.gloss());
			if (p == null || p.speaker != Pronoun.Speaker.THIRD_PERSON) {
				continue;
			}
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
								return null;
							}
							return curMention;
						}
					}
				}
			}
		}
		return null;
	}

	private int hobbs(Mention mention1, Mention mention2) {
		Pronoun p = Pronoun.valueOrNull(mention1.gloss());
		if (p == null || p.speaker != Pronoun.Speaker.THIRD_PERSON) {
			return 0;
		}
		if (mention1.doc.indexOfSentence(mention1.sentence) == mention2.doc
				.indexOfSentence(mention2.sentence)) {
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
						if (result && curMention.equals(mention2)) {
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
							if (curMention.equals(mention2)) {
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

	private Mention hobbsMatch(HashSet<Mention> m1) {

		if (m1.size() == 0) {
			return null;
		}

		for (Mention mention1 : m1) {
			Pronoun p = Pronoun.valueOrNull(mention1.gloss());
			if (p == null || p.speaker != Pronoun.Speaker.THIRD_PERSON) {
				continue;
			}
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
						if (result
								&& p.plural == curMention.headToken()
										.isPluralNoun()) {
							return curMention;
						} else {
							return null;
						}
					}
				}
			}
		}
		return null;
	}

	private boolean exactMatch(HashSet<Mention> m1, HashSet<Mention> m2) {
		if (m1.equals(m2)) {
			return false;
		}
		if (m1.size() == 0 || m2.size() == 0) {
			return false;
		}

		List<Mention> m1copy = new ArrayList<Mention>(m1);
		List<Mention> m2copy = new ArrayList<Mention>(m2);

		String s1 = m1copy.get(0).gloss();
		String s2 = m2copy.get(0).gloss();
		if (Pronoun.isSomePronoun(s1)) {
			return false;
		}

		if (s1.equalsIgnoreCase(s2)) {
			merge(m1, m2);
			return true;
		}

		return false;
	}

	private void pronounMatch(Set<Mention> cluster,
			Set<HashSet<Mention>> clusters) {
		boolean hasNoun = false;
		Iterator<Mention> iter = cluster.iterator();
		while (iter.hasNext()) {
			Mention next = iter.next();
			if (!Pronoun.isSomePronoun(next.headWord())) {
				hasNoun = true;
				break;
			}
		}
		if (hasNoun) {
			return;
		}

		int min_dis = 10000;
		Set<Mention> min_cluster = null;

		iter = cluster.iterator();
		while (iter.hasNext()) {
			Mention next = iter.next();
			Position next_pos = new Position(next);
			for (HashSet<Mention> nextCluster : clusters) {
				if (cluster.equals(nextCluster)) {
					continue;
				}
				for (Mention match : nextCluster) {
					Pronoun p_match = Pronoun.valueOrNull(match.headWord());
					Pronoun p_next = Pronoun.valueOrNull(next.headWord());
					if (p_match == null || p_next == null) {
						continue;
					}
					if (p_match.speaker == p_next.speaker
							&& p_match.plural == p_next.plural) {
						Pair<Boolean, Boolean> gender = Util
								.haveGenderAndAreSameGender(match, next);
						if (gender.getFirst() && gender.getSecond()) {
							Position match_pos = new Position(match);
							int dis = next_pos.distance(match_pos);
							if (dis > 0 && dis < min_dis) {
								min_dis = dis;
								min_cluster = nextCluster;
							}
						}
					}

				}
			}
		}

		if (min_cluster != null) {
			merge(min_cluster, cluster);
		}

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {

		clusters = new HashSet<HashSet<Mention>>();
		for (Mention m : doc.getMentions()) {
			HashSet<Mention> tmp = new HashSet<Mention>();
			tmp.add(m);
			clusters.add(tmp);
		}

		// exact match
		for (HashSet<Mention> m1 : clusters) {
			for (HashSet<Mention> m2 : clusters) {
				if (exactMatch(m1, m2)) {
					break;
				}
			}
		}

		for (Set<Mention> m1 : clusters) {
			for (Set<Mention> m2 : clusters) {
				if (!m1.equals(m2)) {
					if (isAppositive(m1, m2)) {
						merge(m1, m2);
						break;
					}
				}
			}
		}

		for (Set<Mention> m1 : clusters) {
			for (Set<Mention> m2 : clusters) {
				if (!m1.equals(m2)) {
					if (isSameHead(m1, m2)) {
						merge(m1, m2);
						break;
					}
				}
			}
		}

		// hobbs
		for (HashSet<Mention> m1 : clusters) {
			Mention toAdd = hobbsMatch(m1);
			if (toAdd != null) {
				for (HashSet<Mention> m2 : clusters) {
					if (!m1.equals(m2) && m2.contains(toAdd)) {
						merge(m1, m2);
					}
				}
			}
		}

		for (HashSet<Mention> m1 : clusters) {
			Mention toAdd = hobbsPreSentence(m1);
			if (toAdd != null) {
				for (HashSet<Mention> m2 : clusters) {
					if (!m1.equals(m2) && m2.contains(toAdd)) {
						merge(m1, m2);
					}
				}
			}
		}

		for (HashSet<Mention> cluster : clusters) {
			pronounMatch(cluster, clusters);
		}

		ArrayList<Pair<HashSet<Mention>, HashSet<Mention>>> toMerge = new ArrayList<Pair<HashSet<Mention>, HashSet<Mention>>>();
		for (HashSet<Mention> m1 : clusters) {
			for (Mention mention1 : m1) {
				for (HashSet<Mention> m2 : clusters) {
					if (m1.equals(m2)) {
						continue;
					}
					boolean merge = false;
					for (Mention mention2 : m2) {
						if (mention1.headWord().equalsIgnoreCase(
								mention2.headWord())) {
							int index1 = mention1.doc.indexOfMention(mention1);
							int index2 = mention1.doc.indexOfMention(mention2);
							if(Math.abs(index1-index2)<30){
								toMerge.add(new Pair<HashSet<Mention>, HashSet<Mention>>(
										m1, m2));
								break;
							}
						}
					}
					if (merge) {
						break;
					}
				}
			}
		}

		for (int i = 0; i < toMerge.size(); i++) {
			Pair<HashSet<Mention>, HashSet<Mention>> todo = toMerge.get(i);
			merge(todo.getFirst(), todo.getSecond());
		}

		for (Set<Mention> a : clusters) {
			for (Set<Mention> b : clusters) {
				if (a.equals(b)) {
					continue;
				}
				if (sameModifiers(a, b)) {
					merge(a, b);
					break;
				}
			}
		}

		int one = 0;
		int more = 0;

		List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
		for (HashSet<Mention> cluster : clusters) {
			if (cluster.size() == 0) {
				continue;
			}
			if (cluster.size() == 1) {
				one++;
			} else {
				more++;
			}
			ClusteredMention cm = null;
			for (Mention m : cluster) {
				if (cm != null)
					mentions.add(m.markCoreferent(cm));
				else {
					cm = m.markSingleton();
					mentions.add(cm);
				}
			}
		}
//
//
//		 System.out.println(clusters);
//		 System.out.println(doc.sentences);
//		 System.out.println();

		return mentions;
	}

	private HashSet<HashSet<Mention>> clusters;
}
