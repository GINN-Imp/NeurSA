import org.apache.log4j.Level;
import spoon.processing.AbstractProcessor;
import spoon.reflect.code.CtCatch;
import spoon.reflect.cu.SourcePosition;
import spoon.reflect.declaration.CtClass;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.declaration.CtTypedElement;

import java.util.*;

/**
 * Reports warnings when empty catch blocks are found.
 */
public class fileLocProcessor extends AbstractProcessor<CtMethod> {
	public CtMethod method = null;
	public List<BuggyInfo> buggyInfo;
	public Map<BuggyInfo, CtMethod> buggyInfoMap = new HashMap<>();
	public Map<BuggyInfo, Integer> lineInfo = new HashMap<>();

	public Map<CtMethod, BuggyInfo> getBuggyInfo() {
		lineInfo.clear();
		buggyInfo.clear();

		return swapMap();
	}

	private Map<CtMethod, BuggyInfo> swapMap() {
		Map<CtMethod, BuggyInfo > newmap = new HashMap<>();

		for (Map.Entry<BuggyInfo, CtMethod> entry : buggyInfoMap.entrySet()) {
			CtMethod oldVal = entry.getValue();
			BuggyInfo oldKey = entry.getKey();

			newmap.put(oldVal, oldKey);
		}
		return newmap;
	}

	fileLocProcessor(List<BuggyInfo> buggyInfo) {
		if (buggyInfo != null) {
		    this.buggyInfo = buggyInfo;
		    for (BuggyInfo bi:buggyInfo) {
				lineInfo.put(bi, 0);
			}
		}
	}

	@Override
	public boolean isToBeProcessed(CtMethod candidate) {
		return true;
	}

	@Override
	public void process(CtMethod element) {
		SourcePosition position = element.getPosition();
		if(position.isValidPosition()) {
			for (BuggyInfo bi : buggyInfo) {
				if (! position.getFile().toString().endsWith(bi.targetFile)) {
					continue;
				}
				int methodLine = position.getLine();
				if (methodLine > lineInfo.get(bi) && methodLine < bi.targetLine) {
					lineInfo.put(bi, methodLine);
					buggyInfoMap.put(bi, element);
				}
			}
		}
	}
	/*
	@Override
	public void process(CtClass element) {
		SourcePosition position = element.getPosition();
		if(position.isValidPosition()) {
			if (! position.getFile().toString().endsWith(targetFile)) {
			    return;
			}
			Set<CtMethod> methodSet = element.getAllMethods();
			for (CtMethod method : methodSet){
				SourcePosition methodPosition = method.getPosition();
				if(!methodPosition.isValidPosition()) continue;
				int methodLine = methodPosition.getLine();
				if (methodLine > curLine && methodLine < targetLine) {
					curLine = methodLine;
					this.method = method;
				}
			}
		}
	}
	*/
}