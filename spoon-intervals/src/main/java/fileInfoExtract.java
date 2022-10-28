import spoon.processing.AbstractProcessor;
import spoon.reflect.code.CtBlock;
import spoon.reflect.code.CtStatement;
import spoon.reflect.cu.SourcePosition;
import spoon.reflect.cu.position.NoSourcePosition;
import spoon.reflect.declaration.CtMethod;
import spoon.support.sniper.internal.ElementSourceFragment;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Reports warnings when empty catch blocks are found.
 */
public class fileInfoExtract extends AbstractProcessor<CtMethod> {

	public List<Integer> lineInfo = new ArrayList<>();

	fileInfoExtract() {
	}

	private double calculateAverage(List <Integer> marks) {
		Integer sum = 0;
		if(!marks.isEmpty()) {
			for (Integer mark : marks) {
				sum += mark;
			}
			return sum.doubleValue() / marks.size();
		}
		return sum;
	}

	public double getLineInfo(String path, String projName) {

		String collect = new String();
		collect = path;
		for (int i : lineInfo) {
			collect +=  "," + i;
		}
		collect += "\n";
		String outputFileName = projName+".csv";

		try {
			FileWriter writer = new FileWriter(outputFileName, true);
			writer.write(collect);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return calculateAverage(lineInfo);
	}

	@Override
	public boolean isToBeProcessed(CtMethod candidate) {
		return true;
	}

	@Override
	public void process(CtMethod element) {
		CtBlock body = element.getBody();
		if (body == null)
			return;

		//System.out.println(element.getSimpleName());
		SourcePosition sp = element.getPosition();
		if (sp instanceof NoSourcePosition)
			return;
		int totalLineOfMethod = sp.getEndLine()-sp.getLine()+1;
		lineInfo.add(totalLineOfMethod);
		//ElementSourceFragment ele = element.getOriginalSourceFragment();
		//System.out.println(body.getStatements().size());
		//System.out.println(ele.getSourceCode());
		//System.out.println(ele.getStart());
		//System.out.println(ele.getEnd());
		//System.out.println("00000000");

	}
}