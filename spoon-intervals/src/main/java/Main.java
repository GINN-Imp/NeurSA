
import spoon.Launcher;
import spoon.processing.ProcessingManager;
import spoon.reflect.factory.Factory;
import spoon.support.QueueProcessingManager;
import spoon.reflect.declaration.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.apache.commons.cli.*;



public class Main {
    public  static List<BuggyInfo> parseBuggyFile(String fileName) {
		BufferedReader reader;
		String[] res = null;
		List<BuggyInfo> ret = new ArrayList<>();
        if (fileName == null){
            return null;
        }
		try {
			reader = new BufferedReader(new FileReader(fileName));

			String line;
			while ((line = reader.readLine()) != null) {
			    ret.add(new BuggyInfo(line));
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return ret;
	}

	public static Map<CtMethod, BuggyInfo> returnBuggyLoc(List<BuggyInfo> buggyInfo, final Factory factory) {
        if (buggyInfo == null) {
            return null;
        }
		final ProcessingManager processingManager = new QueueProcessingManager(factory);
		final fileLocProcessor processor = new fileLocProcessor(buggyInfo);
		processingManager.addProcessor(processor);
		processingManager.process(factory.Class().getAll());
		return processor.getBuggyInfo();
	}

	public static CommandLine parseArg(String [] args) {
    	Options options = new Options();

		Option input = new Option("d", "dir", true, "input file or dir");
		input.setRequired(true);
		options.addOption(input);

		Option output = new Option("o", "output", true, "output json files");
		output.setRequired(true);
		options.addOption(output);

		Option buggyFile = new Option("b", "buggyfile", true, "buggy file name");
		buggyFile.setRequired(false);
		options.addOption(buggyFile);


		Option dumpCFG = new Option("CFG", "dumpCFG", false, "dump CFG instead of intervals");
		dumpCFG.setRequired(false);
		options.addOption(dumpCFG);

		Option dumpAST = new Option("AST", "dumpAST", false, "dump AST with DFA");
		dumpAST.setRequired(false);
		options.addOption(dumpAST);

		Option isBuggy = new Option("fix", "isfix", false,
				"if it is a buggy or fix (default: buggy)");
		isBuggy.setRequired(false);
		options.addOption(isBuggy);

		Option outputclean = new Option("clean", "useclean", false,
				"output random clean methods in the same file. (default: false)");
		outputclean.setRequired(false);
		options.addOption(outputclean);

		Option outputCleanAll = new Option("cleanall", "usecleanall", false,
				"output all clean methods in the same file. (default: false)");
		outputCleanAll.setRequired(false);
		options.addOption(outputCleanAll);

		CommandLineParser parser = new DefaultParser();
		HelpFormatter formatter = new HelpFormatter();
		CommandLine cmd = null;

		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			System.out.println(e.getMessage());
			formatter.printHelp("toolname", options);

			System.exit(1);
		}
		return cmd;
	}

	public static void main(String [] args) {
		final String[] myArg = {
				//"-i", "/Users/fff000/Downloads/jfreechart-1.0.19",
				"-i", "src/test/java/GGNNTest",
				"-o", "target/spooned/",
		};

		CommandLine cmd = parseArg(args);

		String buggyLines = null;
		String outputXML = "";
		boolean isDumpCFG = false;
		boolean isDumpAST = false;
		boolean isFix = false;
		boolean outputClean = false;
		boolean outputCleanAll = false;
		myArg[1] = cmd.getOptionValue("dir");
		outputXML = cmd.getOptionValue("output");
		buggyLines = cmd.getOptionValue("buggyfile");
		if (cmd.hasOption("dumpCFG"))
			isDumpCFG = true;
		if (cmd.hasOption("dumpAST"))
			isDumpAST = true;
		if (cmd.hasOption("isfix"))
			isFix = true;
		if (cmd.hasOption("useclean"))
			outputClean = true;
		if (cmd.hasOption("usecleanall"))
			outputCleanAll = true;

		List<BuggyInfo> buggyInfo = parseBuggyFile(buggyLines);

		final Launcher launcher = new Launcher();
		launcher.setArgs(myArg);
		//launcher.run();
		launcher.buildModel();


		int i = 0;
		final Factory factory = launcher.getFactory();

		Map<CtMethod, BuggyInfo> buggyInfoMap = returnBuggyLoc(buggyInfo, factory);

		CFGInt splitedCFGs = new CFGInt();
		for (CtType<?> type : factory.Class().getAll()) {
			if (type.isInterface() || type.isAnnotationType())
				continue;
			CtClass clazz = (CtClass) type;
			Set<CtMethod> methodSet = clazz.getAllMethods();
			for (CtMethod method : methodSet) {
				//cfgConstruct(method);
				if(method.isShadow())
					continue;
				if (method == null || method.getBody() == null) {
					continue;
				}
				splitedCFGs.addGraphs(method, buggyInfoMap, isFix, outputClean, outputCleanAll);
				/*
				final List<CtInvocation<?>> elements = method.getElements(new AbstractFilter<CtInvocation<?>>() {
					@Override
					public boolean matches(CtInvocation<?> element) {
						return super.matches(element);
					}
				});
				for (Iterator<CtInvocation<?>> it = elements.iterator(); it.hasNext(); ) {
					CtInvocation element = it.next();
					System.out.println(i + ". " + clazz.getSimpleName() +
							" - " + method.getSimpleName() + " = " + element.getExecutable().toString());
					i++;
				}
				*/
			}
		}

		splitedCFGs.dumpJson(outputXML, isDumpCFG, isDumpAST);

	}
}
