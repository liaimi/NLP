package cs224n.assignment;

import cs224n.ling.Tree;
import java.util.*;

/**
 * The CKY PCFG Parser you will implement.
 */
public class PCFGParser implements Parser {
    private Grammar grammar;
    private Lexicon lexicon;

    public void train(List<Tree<String>> trainTrees) {
        // TODO: before you generate your grammar, the training trees
        // need to be binarized so that rules are at most binary
        lexicon = new Lexicon(trainTrees);
        grammar = new Grammar(trainTrees);
    }

    public Tree<String> getBestParse(List<String> sentence) {
        // TODO: implement this method
        return null;
    }
}
