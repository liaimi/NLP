package cs224n.corefsystems;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.coref.Mention;
import cs224n.util.Pair;

public class OneCluster implements CoreferenceSystem {

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// TODO Auto-generated method stub

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
	    //(variables)
		Entity ent = new Entity(doc.getMentions());
	    List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();

	    for(Mention m : doc.getMentions()){
	    	ClusteredMention newCluster = m.markCoreferent(ent);
	        mentions.add(newCluster);
	    }
	    //(return the mentions)
	    return mentions;
	}

}
