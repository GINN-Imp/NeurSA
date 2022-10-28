import spoon.reflect.code.CtInvocation;
import spoon.reflect.reference.CtExecutableReference;
import spoon.reflect.visitor.Filter;

public class InvocationFilter  implements Filter<CtInvocation<?>> {

    private CtExecutableReference<?> executable;

    public InvocationFilter(CtExecutableReference<?> executable) {
        this.executable = executable;
    }

    @Override
    public boolean matches(CtInvocation<?> invocation) {
        return invocation.getExecutable().equals(executable);
    }
}