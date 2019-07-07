import org.apache.commons.collections4.bag.SynchronizedSortedBag;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;
import spoon.reflect.path.CtRole;
import spoon.reflect.visitor.CtScanner;

import java.util.ArrayList;
import java.util.List;

public class tokenVisitor extends CtScanner {
    public String label = "";
    public int [] vector = new int[tokenIndex.Size];
    private List<Integer> tokenSeq = new ArrayList<Integer>();
    boolean ifProcess = false;

    public int[] getVector() {
        return vector;
    }

    public int[] getTokenSeq() {
        int[] ret = new int[tokenSeq.size()];
        int i = 0;
        for (Integer e : tokenSeq)
            ret[i++] = e;
        return ret;
    }

    public void updateVector(int index, CtElement element) {
        if (element != null && ifProcess(element) == false)
            return;
        vector[index]++;
        tokenSeq.add(index);
    }

    private boolean ifProcess(CtElement element) {
        if (element.getDirectChildren().size() > 1) {
            return false;
        }
        return true;
    }

    @Override
    public void enter(CtElement element) {
    }

    @Override
    public void exit(CtElement element) {
    }

    @Override
    public void visitCtAnonymousExecutable(CtAnonymousExecutable anonymousExec) {
        updateVector(tokenIndex.AnonymousExecutable, anonymousExec);
        super.visitCtAnonymousExecutable(anonymousExec);
    }

    @Override
    public <T> void visitCtArrayRead(CtArrayRead<T> arrayRead) {
        updateVector(tokenIndex.ArrayRead, arrayRead);
        super.visitCtArrayRead(arrayRead);
    }

    @Override
    public <T> void visitCtArrayWrite(CtArrayWrite<T> arrayWrite) {
        updateVector(tokenIndex.ArrayWrite, arrayWrite);
        super.visitCtArrayWrite(arrayWrite);
    }

    public <T> void visitCtArrayTypeReference(CtArrayTypeReference<T> reference) {
        updateVector(tokenIndex.ArrayTypeReference, reference);
        //super.visitCtArrayTypeReference(reference);
    }

    public <T> void visitCtAssert(CtAssert<T> asserted) {
        updateVector(tokenIndex.Assert, asserted);
        super.visitCtAssert(asserted);
    }

    public <T, A extends T> void visitCtAssignment(
            CtAssignment<T, A> assignement) {
        updateVector(tokenIndex.Assignment, assignement);
        super.visitCtAssignment(assignement);
    }

    public <T> void visitCtBinaryOperator(CtBinaryOperator<T> operator) {
        //vector[tokenIndex.BinaryOperator] += operator.getKind().ordinal();
        updateVector(tokenIndex.BinaryOperator, operator);
        super.visitCtBinaryOperator(operator);
    }

    public <R> void visitCtBlock(CtBlock<R> block) {
        updateVector(tokenIndex.Block, block);
        super.visitCtBlock(block);
    }

    public void visitCtBreak(CtBreak breakStatement) {
        updateVector(tokenIndex.Break, breakStatement);
        super.visitCtBreak(breakStatement);
    }

    public <S> void visitCtCase(CtCase<S> caseStatement) {
        updateVector(tokenIndex.Case, caseStatement);
        super.visitCtCase(caseStatement);
    }

    public void visitCtCatch(CtCatch catchBlock) {
        updateVector(tokenIndex.Catch, catchBlock);
        super.visitCtCatch(catchBlock);
    }

    public <T> void visitCtClass(CtClass<T> ctClass) {
        // don't think it's reachable
        super.visitCtClass(ctClass);
    }

    public <T> void visitCtConditional(CtConditional<T> conditional) {
        updateVector(tokenIndex.Conditional, conditional);
        super.visitCtConditional(conditional);
    }

    public <T> void visitCtConstructor(CtConstructor<T> c) {
        updateVector(tokenIndex.Constructor, c);
        super.visitCtConstructor(c);
    }

    public void visitCtContinue(CtContinue continueStatement) {
        updateVector(tokenIndex.Continue, continueStatement);
        super.visitCtContinue(continueStatement);
    }

    public void visitCtDo(CtDo doLoop) {
        updateVector(tokenIndex.Do, doLoop);
        super.visitCtDo(doLoop);
    }

    public <T extends Enum<?>> void visitCtEnum(CtEnum<T> ctEnum) {
        updateVector(tokenIndex.Enum, ctEnum);
        super.visitCtEnum(ctEnum);
    }

    public <T> void visitCtExecutableReference(
            CtExecutableReference<T> reference) {
        updateVector(tokenIndex.ExecutableReference, reference);
        super.visitCtExecutableReference(reference);
    }

    public <T> void visitCtField(CtField<T> f) {
        updateVector(tokenIndex.Field, f);
        super.visitCtField(f);
    }

    @Override
    public <T> void visitCtThisAccess(CtThisAccess<T> thisAccess) {
        updateVector(tokenIndex.ThisAccess, thisAccess);
        super.visitCtThisAccess(thisAccess);
    }

    public <T> void visitCtAnnotationFieldAccess(
            CtAnnotationFieldAccess<T> annotationFieldAccess) {
        updateVector(tokenIndex.AnnotationFieldAccess, annotationFieldAccess);
        super.visitCtAnnotationFieldAccess(annotationFieldAccess);
    }

    public <T> void visitCtFieldReference(CtFieldReference<T> reference) {
        updateVector(tokenIndex.FieldReference, reference);
        super.visitCtFieldReference(reference);
    }

    public void visitCtFor(CtFor forLoop) {
        updateVector(tokenIndex.For, forLoop);
        super.visitCtFor(forLoop);
    }

    public void visitCtForEach(CtForEach foreach) {
        updateVector(tokenIndex.ForEach, foreach);
        super.visitCtForEach(foreach);
    }

    public void visitCtIf(CtIf ifElement) {
        updateVector(tokenIndex.If, ifElement);
        super.visitCtIf(ifElement);
    }

    public <T> void visitCtInterface(CtInterface<T> intrface) {
        updateVector(tokenIndex.Interface, intrface);
        super.visitCtInterface(intrface);
    }

    public <T> void visitCtInvocation(CtInvocation<T> invocation) {
        updateVector(tokenIndex.Invocation, invocation);
        super.visitCtInvocation(invocation);
    }

    public <T> void visitCtLiteral(CtLiteral<T> literal) {
        updateVector(tokenIndex.Literal, literal);
        super.visitCtLiteral(literal);
    }

    public <T> void visitCtLocalVariable(CtLocalVariable<T> localVariable) {
        updateVector(tokenIndex.LocalVariable, localVariable);
        super.visitCtLocalVariable(localVariable);
    }

    public <T> void visitCtLocalVariableReference(
            CtLocalVariableReference<T> reference) {
        updateVector(tokenIndex.LocalVariableReference, reference);
        super.visitCtLocalVariableReference(reference);
    }

    public <T> void visitCtCatchVariable(CtCatchVariable<T> catchVariable) {
        updateVector(tokenIndex.CatchVariable, catchVariable);
        super.visitCtCatchVariable(catchVariable);
    }

    public <T> void visitCtCatchVariableReference(CtCatchVariableReference<T> reference) {
        updateVector(tokenIndex.CatchVariableReference, reference);
        super.visitCtCatchVariableReference(reference);
    }

    public <T> void visitCtMethod(CtMethod<T> m) {
        updateVector(tokenIndex.Method, m);
        super.visitCtMethod(m);
    }

    public <T> void visitCtNewArray(CtNewArray<T> newArray) {
        updateVector(tokenIndex.NewArray, newArray);
        super.visitCtNewArray(newArray);
    }

    @Override
    public <T> void visitCtConstructorCall(CtConstructorCall<T> ctConstructorCall) {
        updateVector(tokenIndex.ConstructorCall, ctConstructorCall);
        super.visitCtConstructorCall(ctConstructorCall);
    }

    public <T> void visitCtNewClass(CtNewClass<T> newClass) {
        updateVector(tokenIndex.NewClass, newClass);
        super.visitCtNewClass(newClass);
    }

    @Override
    public <T> void visitCtLambda(CtLambda<T> lambda) {
        updateVector(tokenIndex.Lambda, lambda);
        super.visitCtLambda(lambda);
    }

    @Override
    public <T, E extends CtExpression<?>> void visitCtExecutableReferenceExpression(
            CtExecutableReferenceExpression<T, E> expression) {
        updateVector(tokenIndex.ExecutableReferenceExpression, expression);
        super.visitCtExecutableReferenceExpression(expression);
    }

    public <T, A extends T> void visitCtOperatorAssignment(
            CtOperatorAssignment<T, A> assignment) {
        updateVector(tokenIndex.OperatorAssignment, assignment);
        super.visitCtOperatorAssignment(assignment);
    }

    public void visitCtPackage(CtPackage ctPackage) {
        updateVector(tokenIndex.Package, ctPackage);
        super.visitCtPackage(ctPackage);
    }

    public void visitCtPackageReference(CtPackageReference reference) {
        updateVector(tokenIndex.PackageReference, reference);
        super.visitCtPackageReference(reference);
    }

    public <T> void visitCtParameter(CtParameter<T> parameter) {
        updateVector(tokenIndex.Parameter, parameter);
        super.visitCtParameter(parameter);
    }

    public <T> void visitCtParameterReference(CtParameterReference<T> reference) {
        updateVector(tokenIndex.ParameterReference, reference);
        super.visitCtParameterReference(reference);
    }

    public <R> void visitCtReturn(CtReturn<R> returnStatement) {
        updateVector(tokenIndex.Return, returnStatement);
        super.visitCtReturn(returnStatement);
    }

    public <R> void visitCtStatementList(CtStatementList statements) {
        updateVector(tokenIndex.StatementList, statements);
        super.visitCtStatementList(statements);
    }

    public <S> void visitCtSwitch(CtSwitch<S> switchStatement) {
        updateVector(tokenIndex.Switch, switchStatement);
        super.visitCtSwitch(switchStatement);
    }

    public void visitCtSynchronized(CtSynchronized synchro) {
        updateVector(tokenIndex.Synchronized, synchro);
        super.visitCtSynchronized(synchro);
    }

    public void visitCtThrow(CtThrow throwStatement) {
        updateVector(tokenIndex.Throw, throwStatement);
        super.visitCtThrow(throwStatement);
    }

    public void visitCtTry(CtTry tryBlock) {
        updateVector(tokenIndex.Try, tryBlock);
        super.visitCtTry(tryBlock);
    }

    @Override
    public void visitCtTryWithResource(CtTryWithResource tryWithResource) {
        updateVector(tokenIndex.TryWithResource, tryWithResource);
        super.visitCtTryWithResource(tryWithResource);
    }

    public void visitCtTypeParameter(CtTypeParameter typeParameter) {
        updateVector(tokenIndex.TypeParameter, typeParameter);
        //super.visitCtTypeParameter(typeParameter);
    }

    public void visitCtTypeParameterReference(CtTypeParameterReference ref) {
        updateVector(tokenIndex.TypeParameterReference, ref);
        //super.visitCtTypeParameterReference(ref);
    }

    private void parseType(CtTypeReference reference) {
        CtTypeReference tr = reference.getTopLevelType();
        if (tr== null) {
            tr = reference;
        }
        if (tr == null || !tr.isPrimitive()) {
            updateVector(tokenIndex.NonPrimitiveType, tr);
        }
        else {
            updateVector(tokenIndex.PrimitiveType, tr);
        }
    }
    public <T> void visitCtTypeReference(CtTypeReference<T> reference) {
        //updateVector(tokenIndex.TypeReference);
        parseType(reference);
        //super.visitCtTypeReference(reference);
    }

    @Override
    public <T> void visitCtTypeAccess(CtTypeAccess<T> typeAccess) {
        updateVector(tokenIndex.TypeAccess, typeAccess);
        super.visitCtTypeAccess(typeAccess);
    }

    public <T> void visitCtUnaryOperator(CtUnaryOperator<T> operator) {
        updateVector(tokenIndex.UnaryOperator, operator);
        super.visitCtUnaryOperator(operator);
    }

    public <T> void visitCtVariableAccess(CtVariableAccess<T> variableAccess) {
        updateVector(tokenIndex.VariableAccess, variableAccess);
        //super.visitCtVariableAccess(variableAccess);
    }

    public <T> void visitCtVariableRead(CtVariableRead<T> variableRead) {
        updateVector(tokenIndex.VariableRead, variableRead);
        super.visitCtVariableRead(variableRead);
    }

    @Override
    public <T> void visitCtVariableWrite(CtVariableWrite<T> variableWrite) {
        updateVector(tokenIndex.VariableWrite, variableWrite);
        super.visitCtVariableWrite(variableWrite);
    }

    public void visitCtWhile(CtWhile whileLoop) {
        updateVector(tokenIndex.While, whileLoop);
        super.visitCtWhile(whileLoop);
    }

    public <T> void visitCtCodeSnippetExpression(
            CtCodeSnippetExpression<T> expression) {
        updateVector(tokenIndex.CodeSnippetExpression, expression);
    }

    public void visitCtCodeSnippetStatement(CtCodeSnippetStatement statement) {
        updateVector(tokenIndex.CodeSnippetStatement, statement);
    }

    public <T> void visitCtUnboundVariableReference(
            CtUnboundVariableReference<T> reference) {
        updateVector(tokenIndex.UnboundVariableReference, reference);

    }

    public <T> void visitCtFieldAccess(CtFieldAccess<T> f) {
        updateVector(tokenIndex.FieldAccess, f);
    }

    @Override
    public <T> void visitCtFieldRead(CtFieldRead<T> fieldRead) {
        updateVector(tokenIndex.FieldRead, fieldRead);
        visitCtFieldAccess(fieldRead);
    }

    @Override
    public <T> void visitCtFieldWrite(CtFieldWrite<T> fieldWrite) {
        updateVector(tokenIndex.FieldWrite, fieldWrite);
        visitCtFieldAccess(fieldWrite);
    }

    @Override
    public <T> void visitCtSuperAccess(CtSuperAccess<T> f) {
        updateVector(tokenIndex.SuperAccess, f);
        super.visitCtSuperAccess(f);
    }

}
