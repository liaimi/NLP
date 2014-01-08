package cs224n.corefsystems;

import java.util.ArrayList;
import java.util.Iterator;

import cs224n.coref.Document;
import cs224n.coref.Mention;
import cs224n.ling.Tree;
import cs224n.util.Pair;

public class HobbsHelper {

	private static ArrayList<Integer> SameSentenceHoobs(Mention mention) {
		ArrayList<Integer> pos = new ArrayList<Integer>();

		for (int i = 0; i < mention.beginIndexInclusive; i++) {

			Iterable<Pair<String, Integer>> output = mention.sentence.parse
					.getTraversalBetween(mention.beginIndexInclusive, i);
			Iterator<Pair<String, Integer>> iter = output.iterator();

			boolean notCurSentence = false;
			boolean notThis = false;
			int NP = 0;
			boolean metX = false;
			boolean metNP = false;

			while (iter.hasNext()) {
				Pair<String, Integer> cur = iter.next();
				if (cur.getSecond() == 1) {
					notThis = true;
					break;
				} else if ((cur.getFirst().matches("NP"))) {
					metNP = true;
					break;
				}
			}

			while (iter.hasNext()) {
				Pair<String, Integer> cur = iter.next();

				if ((cur.getFirst().matches("NP") || cur.getFirst()
						.matches("S"))) {
					if (cur.getSecond() == 0) {
						metX = true;
						break;
					} else {
						notThis = true;
						break;
					}
				} else if (cur.getSecond() == 1) {
					notCurSentence = true;
					break;
				}
			}

			while (iter.hasNext()) {
				Pair<String, Integer> cur = iter.next();
				if (cur.getSecond() == -1) {
					break;
				} else if ((cur.getFirst().matches("NP") || cur.getFirst()
						.matches("S"))) {
					NP++;
				}
			}

			if (metNP && NP == 2 && metX && !iter.hasNext()) {
				pos.add(i);
			}

		}

		return pos;
	}

	public static boolean top(Mention mention) {

		if (mention.beginIndexInclusive == 0) {
			return true;
		}
		Iterable<Pair<String, Integer>> output = mention.sentence.parse
				.getTraversalBetween( mention.beginIndexInclusive, 0);

		Iterator<Pair<String, Integer>> iter = output.iterator();

		int np = 0;
		while (iter.hasNext()) {
			Pair<String, Integer> cur = iter.next();
			if (cur.getFirst().equalsIgnoreCase("NP")
					|| cur.getFirst().equalsIgnoreCase("S")) {
				np++;
			}
			if (cur.getSecond() == 1) {
				break;
			}
		}
		if (np <= 1) {
			return true;
		}
		return false;
	}

	public static boolean getSameSentence(Mention mention, Mention mention2) {
		if (mention.doc.indexOfSentence(mention.sentence) == mention.doc
				.indexOfSentence(mention2.sentence)) {
			ArrayList<Integer> pos = SameSentenceHoobs(mention);
			if (pos.size() != 0) {
				for (int i = 0; i < pos.size(); i++) {
					int position = pos.get(i);
					if (position >= mention2.beginIndexInclusive
							&& position < mention2.endIndexExclusive) {
						return true;
					}
				}
			}
		}
		return false;
	}
}

class Position {
	public int sentenceIndex;
	public int wordIndexStart;
	public int wordIndexEnd;
	public Document doc;
	public Mention mention;

	public Position(int sentenceIndex, int wordIndexStart, int wordIndexEnd,
			Document doc) {
		this.sentenceIndex = sentenceIndex;
		this.wordIndexStart = wordIndexStart;
		this.wordIndexEnd = wordIndexEnd;
		this.doc = doc;
	}

	public Position(Mention mention) {
		this.mention = mention;
		this.doc = mention.doc;
		this.sentenceIndex = doc.indexOfSentence(mention.sentence);
		this.wordIndexStart = mention.beginIndexInclusive;
		this.wordIndexEnd = mention.endIndexExclusive;
	}

	public int distance(Position pos) {
		if (sentenceIndex > pos.sentenceIndex) {
			return -1;
		}
		if (sentenceIndex == pos.sentenceIndex
				&& wordIndexStart > pos.wordIndexStart) {
			return -1;
		}
		int dis = 0;
		if (this.sentenceIndex == pos.sentenceIndex) {
			return pos.wordIndexStart - this.wordIndexStart;
		}

		for (int i = this.sentenceIndex + 1; i < pos.sentenceIndex; i++) {
			dis += doc.sentences.get(i).length();
		}

		dis += mention.sentence.length() - wordIndexStart;
		dis += pos.wordIndexStart;

		return dis;
	}
}