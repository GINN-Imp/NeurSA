import com.github.difflib.UnifiedDiffUtils;
import com.github.difflib.text.DiffRow;
import com.github.difflib.text.DiffRowGenerator;
import gumtree.spoon.diff.Diff;
import org.apache.commons.cli.*;
import spoon.Launcher;
import spoon.reflect.CtModel;
import spoon.reflect.code.CtLoop;
import spoon.reflect.cu.SourcePosition;
import spoon.reflect.cu.position.NoSourcePosition;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.visitor.filter.TypeFilter;

import java.io.File;
import java.io.IOException;
import java.util.*;

import com.github.difflib.DiffUtils;
import com.github.difflib.patch.AbstractDelta;
import com.github.difflib.patch.Patch;

class methods {
    List<Integer> a;
    public methods(List<Integer> c) {
        a = c;
        if (a.size() == 0)
            a.add(-1);
        Collections.sort(a);
    }
    public int sum (){
        if (a.size() > 0) {
            int sum = 0;

            for (Integer i : a) {
                sum += i;
            }
            return sum;
        }
        return 0;
    }
    public double mean (){
        int sum = sum();
        double mean = 0;
        mean = sum / (a.size() * 1.0);
        return mean;
    }
    public double median (){
        int middle = a.size()/2;

        if (a.size() % 2 == 1) {
            return a.get(middle);
        } else {
            return (a.get(middle-1) + a.get(middle)) / 2.0;
        }
    }
    public double sd (){
        int sum = 0;
        double mean = mean();
        for (Integer i : a)
            sum += Math.pow((i - mean), 2);
        return Math.sqrt( sum / ( a.size() ) ); // sample
    }
}

class LoopInfo {
    int lineNum;
    int loopNum;
    CtMethod methodName;
    public LoopInfo(int i, int j, CtMethod method) {
        lineNum = i;
        loopNum = j;
        methodName = method;
    }
}

public class AnalyzeProjInfo {
    public static int MAXFILE = 10000;
    public static List<Integer> lineInfo = new ArrayList<>();
    public static List<Integer> loopInfo = new ArrayList<>();
    public static List<CtMethod> methodNameInfo = new ArrayList<>();
    public static int setNum = 0, getNum = 0;
    public static CommandLine parseArg(String[] args) {
        Options options = new Options();

        Option input = new Option("d", "dir", true, "path to input file");
        input.setRequired(true);
        options.addOption(input);

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
    public  static Launcher retLauncher(String [] arg){
        final Launcher launcher = new Launcher();
        launcher.setArgs(arg);
        //launcher.getEnvironment().setPrettyPrinterCreator(() -> new SniperJavaPrettyPrinter(launcher.getEnvironment()));
        return launcher;
    }

    private static void analyzeJavaFile(String inputFileDir) {
        final String[] myArg = {
                "-i", inputFileDir,
                "-o", "tmp",
        };
        Launcher launcher = retLauncher(myArg);
        try {
            launcher.buildModel();
        }
        catch (Exception e){
            return;
        }
        CtModel model = launcher.getModel();
        for (CtMethod method : model.getElements(new TypeFilter<>(CtMethod.class))) {
            SourcePosition sp = method.getPosition();
            if (sp instanceof NoSourcePosition)
                return;
            int totalLineOfMethod = sp.getEndLine()-sp.getLine()+1;
            int loopCount = method.getElements(new TypeFilter<>(CtLoop.class)).size();
            lineInfo.add(totalLineOfMethod);
            loopInfo.add(loopCount);
            methodNameInfo.add(method);

            if (totalLineOfMethod < 6 ) {
                String methodName = method.getSimpleName();
                if (methodName.toLowerCase().startsWith("get")) {
                    getNum ++;
                }
                else if (methodName.toLowerCase().startsWith("set")) {
                    setNum ++;
                }
            }
        }
    }
    private static int countLE(List<Integer> data, int ind) {
        int count = 0;
        for (Integer num : data) {
            if (num <= ind) count++;
        }
        return count;
    }
    private static void printAnalyzedInfo() {
        printPairInfo();
        methods m = new methods(lineInfo);
        System.out.printf("Method size info: #method: %d. mean: %.3f. median: %.2f. sd: %.2f.\n" , lineInfo.size(), m.mean(), m.median(), m.sd());
        int less5Num = countLE(lineInfo, 5);
        int less10Num = countLE(lineInfo, 10);
        double methodNum = lineInfo.size()/100.0;
        System.out.printf("Proj Info: #method <= 5 loc: %d, #method <=10 loc: %d. #getters: %d. #setters: %d.\n",
                less5Num, less10Num, getNum, setNum);
        System.out.printf("Proj Info: %%method <= 5 loc: %.1f%%, %%method <=10 loc: %.1f%%. %%getters: %.1f%%. %%setters: %.1f%%.\n",
                less5Num/methodNum, less10Num/methodNum, getNum/methodNum, setNum/methodNum);

    }
    private static void printPairInfo(){
        List<LoopInfo> words = new ArrayList<>();
        for (int i = 0; i < lineInfo.size(); i++) {
            int lineNum = lineInfo.get(i);
            int loopNum = loopInfo.get(i);
            CtMethod methodName = methodNameInfo.get(i);
            words.add(new LoopInfo(lineNum, loopNum, methodName));
        }

        Collections.sort(words, new Comparator<LoopInfo>() {
            @Override
            public int compare(final LoopInfo o1, final LoopInfo o2) {
                if (o1.lineNum > o2.lineNum) {
                    return 1;
                } else if (o1.lineNum == o2.lineNum) {
                    return 0;
                } else {
                    return -1;
                }
            }
        });
        System.out.println("Loop info");
        int size = words.size();
        double stride = size/10.0;
        for (int i = 0;i < 10; i ++) {
            int min = (int)(i*stride);
            int max = (int)((i+1)*stride);
            List<Integer> loopInfo = new ArrayList<>();
            for (int j = min; j < max ; j++) {
                int loopNum = words.get(j).loopNum;
                loopInfo.add(loopNum);
            }
            methods c = new methods(loopInfo);
            System.out.printf("From %d loc to %d loc: #method is: %d. Avg #Loop: %.3f. median: %.2f. sd: %.3f.\n" , words.get(min).lineNum,
                    words.get(max-1).lineNum,
                    loopInfo.size(), c.mean(), c.median(), c.sd());
        }
    }

    private static void analyzeJavaFiles(String inputFileDir) {
        List<File> files = new ArrayList<>();
        common.listf(inputFileDir, files);
        int count = 0, totalSize = files.size();
        for (File f : files) {
            analyzeJavaFile(f.getAbsolutePath());
            if (count % 500 == 0) {
                System.out.println("Analyzing " + count + " files in " + inputFileDir + ".");
            }
            if (count > MAXFILE)
                break;
            count++;
        }
    }
    public static int analyzeTwoMethods(CtMethod m1, CtMethod m2) {
        String method1Name = m1.getSimpleName();
        String method2Name = m2.getSimpleName();
        List<String> original = Arrays.asList(m1.toString().replace(method1Name, "AAAA").split("\n"));
        List<String> revised = Arrays.asList(m2.toString().replace(method2Name, "AAAA").split("\n"));
        Patch<String> patch = null;
        int count = 0;
        int count2 = 0;
        try {
            patch = DiffUtils.diff(original, revised);
        }
        catch (Exception e) {
            return -1;
        }
        for (AbstractDelta<String> delta : patch.getDeltas()) {
            count += Math.min(delta.getSource().getLines().size(), delta.getTarget().getLines().size());
            for (String s : delta.getTarget().getLines()) {
                count2 += getTokens(s);
            }
        }
        return count2;
    }
    public static int getTokens(String str) {
        StringTokenizer tokenizer = new StringTokenizer(str, " ");
        return tokenizer.countTokens();
    }
    public static int analyzeTwoMethods2(CtMethod m1, CtMethod m2) {
        List<String> original = Arrays.asList(m1.toString().split("\n"));
        List<String> revised = Arrays.asList(m2.toString().split("\n"));
        List<DiffRow> rows = null;
        int count = 0;
        try {
            DiffRowGenerator generator = DiffRowGenerator.create()
                    .showInlineDiffs(true)
                    .inlineDiffByWord(true)
                    .build();
            rows = generator.generateDiffRows(original, revised);
        }
        catch (Exception e) {
            return -1;
        }
        for (DiffRow r : rows) {
            ;
        }
        return count;
    }
    public static void analyzeTokenSequenceDifference() {
        List<Integer> diffInfo = new ArrayList();
        for (int i = 0; i < Math.min(1000, methodNameInfo.size()-1); i ++) {
            CtMethod m1 = methodNameInfo.get(i);
            CtMethod m2 = methodNameInfo.get(i+1);
            int diffNum = analyzeTwoMethods(m1, m2);
            diffInfo.add(diffNum);
        }
        Collections.sort(diffInfo);
        List<Integer> highInfo = diffInfo.subList(0, diffInfo.size()/4);
        List<Integer> lowInfo = diffInfo.subList(diffInfo.size()*3/4, diffInfo.size());
        methods d = new methods(highInfo);
        System.out.println("avg in the first 25%: " + d.mean());
        d = new methods(lowInfo);
        System.out.println("avg in the last 75%: " + d.mean());
        d = new methods(diffInfo);
        System.out.println("avg in the whole: " + d.mean());
    }

    public static void main(String[] args) {
        CommandLine cmd = parseArg(args);
        String inputDir = cmd.getOptionValue("dir");
        analyzeJavaFiles(inputDir);
        printAnalyzedInfo();
        analyzeTokenSequenceDifference();
    }
}
