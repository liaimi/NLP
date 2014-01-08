package cs224n.assignment;

import cs224n.evaluator.EnglishPennTreebankParseEvaluator;
import cs224n.io.PennTreebankReader;
import cs224n.ling.Tree;
import cs224n.ling.Trees;
import cs224n.util.*;

import java.util.*;

/**
 * Harness for PCFG Parser project.
 *
 * @author Dan Klein
 */
public class PCFGParserTester {

	// Longest sentence length that will be tested on.
	private static int MAX_LENGTH = 20;

	private static void testParser(Parser parser, List<Tree<String>> testTrees) {
		EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String> eval = 
				new EnglishPennTreebankParseEvaluator.LabeledConstituentEval<String>
		(Collections.singleton("ROOT"), 
				new HashSet<String>(Arrays.asList(new String[] {"''", "``", ".", ":", ","})));
		for (Tree<String> testTree : testTrees) {
			List<String> testSentence = testTree.getYield();
			if (testSentence.size() > MAX_LENGTH)
				continue;
			Tree<String> guessedTree = parser.getBestParse(testSentence);
			System.out.println("Guess:\n"+Trees.PennTreeRenderer.render(guessedTree));
			System.out.println("Gold:\n"+Trees.PennTreeRenderer.render(testTree));
			eval.evaluate(guessedTree, testTree);
		}
		eval.display(true);
	}

	private static List<Tree<String>> readTrees(String basePath, int low,
			int high) {
		Collection<Tree<String>> trees = PennTreebankReader.readTrees(basePath,
				low, high);
		// normalize trees
		Trees.TreeTransformer<String> treeTransformer = new Trees.StandardTreeNormalizer();
		List<Tree<String>> normalizedTreeList = new ArrayList<Tree<String>>();
		for (Tree<String> tree : trees) {
			Tree<String> normalizedTree = treeTransformer.transformTree(tree);
			// System.out.println(Trees.PennTreeRenderer.render(normalizedTree));
			normalizedTreeList.add(normalizedTree);
		}
		return normalizedTreeList;
	}

	public static void main(String[] args) {

		// set up default options ..............................................
		Map<String, String> options = new HashMap<String, String>();
		options.put("-path",      "/afs/ir/class/cs224n/pa2/data/");
		options.put("-data",      "miniTest");
		options.put("-parser",    "cs224n.assignment.BaselineParser");
		options.put("-maxLength", "20");

		// let command-line options supersede defaults .........................
		options.putAll(CommandLineUtils.simpleCommandLineParser(args));
		System.out.println("PCFGParserTester options:");
		for (Map.Entry<String, String> entry: options.entrySet()) {
			System.out.printf("  %-12s: %s%n", entry.getKey(), entry.getValue());
		}
		System.out.println();

		MAX_LENGTH = Integer.parseInt(options.get("-maxLength"));

		Parser parser;
		try {
			Class<?> parserClass = Class.forName(options.get("-parser"));
			parser = (Parser) parserClass.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		System.out.println("Using parser: " + parser);

		String basePath = options.get("-path");
		String dataSet = options.get("-data");
		if (!basePath.endsWith("/"))
			basePath += "/";
		//basePath += dataSet;
		System.out.println("Data will be loaded from: " + basePath + "\n");

		List<Tree<String>> trainTrees = new ArrayList<Tree<String>>(),
				validationTrees = new ArrayList<Tree<String>>(),
				testTrees = new ArrayList<Tree<String>>();

		if (!basePath.endsWith("/"))
			basePath += "/";
		basePath += dataSet;
		if (dataSet.equals("miniTest")) {
			System.out.print("Loading training trees...");
			trainTrees = readTrees(basePath, 1, 3);
			System.out.println("done.");
			System.out.print("Loading test trees...");
			testTrees = readTrees(basePath, 4, 4);
			System.out.println("done.");
		}
		else if (dataSet.equals("treebank")) {
			System.out.print("Loading training trees...");
			trainTrees = readTrees(basePath, 200, 2199);
			System.out.println("done.");
			System.out.print("Loading validation trees...");
			validationTrees = readTrees(basePath, 2200, 2299);
			System.out.println("done.");
			System.out.print("Loading test trees...");
			testTrees = readTrees(basePath, 2300, 2319);
			System.out.println("done.");
		}
		else {
			throw new RuntimeException("Bad data set mode: "+ dataSet+", use miniTest, or treebank."); 
		}
		parser.train(trainTrees);
		testParser(parser, testTrees);
	}
}
