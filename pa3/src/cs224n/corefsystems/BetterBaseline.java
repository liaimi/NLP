package cs224n.corefsystems;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

import cs224n.coref.*;
import cs224n.util.Pair;

public class BetterBaseline implements CoreferenceSystem {

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// TODO Auto-generated method stub

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {

		Entity I_ent = new Entity(doc.getMentions());
		Entity we_ent = new Entity(doc.getMentions());
		List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
		Map<String, Entity> clusters = new HashMap<String, Entity>();
		// (for each mention...)
		for (Mention m : doc.getMentions()) {
			String mString = m.gloss().toLowerCase();

			String head = m.headWord();
			// (...if we've seen this text before...)
			if (clusters.containsKey(head)) {
				// (...add it to the cluster)
				mentions.add(m.markCoreferent(clusters.get(head)));
			} else {
				// (...else create a new singleton cluster)
				ClusteredMention newCluster = m.markSingleton();
				mentions.add(newCluster);
				clusters.put(head, newCluster.entity);
			}
		}
		// (return the mentions)
		return mentions;
	}

}