package src;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Vector;

public class test1 {
	public static void main(String[] args) {
		if (args[0] == "1")
			m1(1);
		else
			return;
		return;
	}

	public static void m1(int arg) {
		int i = 9;
		while(arg>10) {
			do {
				if(i>1)
					i--;
				else
					i++;
				System.out.print(i);
				i = arg+i;
			}while(i>10);
			arg++;
		}
		i+=1;
		arg++;
		return;
	}

	public boolean test1(Object obj) {
        return obj == null || obj.getClass( ).getName( ).startsWith( "java.lang." );
	}


	public boolean test2(Object obj, Object obj2) {
        boolean res = obj == obj2.getClass().getEnclosingMethod() || obj.getClass( ).getName( ).startsWith( "java.lang." );
        return res;
	}

	HashSet<String> ints = new HashSet<>();
	void addint(String i) {
		this.ints.add(i);
	}

}
