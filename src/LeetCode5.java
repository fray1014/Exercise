import org.junit.Test;
import util.ListNode;

import java.util.*;

public class LeetCode5 {
    @Test
    public void test(){

    }
    public class Solution224{
        public int calculate(String s) {
            Deque<Character> op = new ArrayDeque<>();
            Deque<Integer> num = new ArrayDeque<>();
            String s2 = s.replace(" ","");
            for(int i=0;i<s2.length();i++){
                if(s2.charAt(i)=='('||s2.charAt(i)=='+'||s2.charAt(i)=='-'){
                    op.push(s2.charAt(i));
                }else if(s2.charAt(i)==')'){
                    while(op.peek()!='('){
                        int n1 = num.pop();
                        int n2 = num.pop();
                        Character op1 = op.pop();
                        Character op2 = Optional.ofNullable(op.peek()).orElse('+');
                        if(op2=='-'){
                            if(op1=='+'){
                                num.push(n2-n1);
                            }else{
                                num.push(n1+n2);
                            }
                        }else{
                            if(op1=='+'){
                                num.push(n1+n2);
                            }else{
                                num.push(n2-n1);
                            }
                        }
                    }
                    op.pop();
                }else{
                    num.push(s2.charAt(i)-'0');
                }
            }
            while(!op.isEmpty()){
                int n1 = num.pop();
                int n2 = num.pop();
                Character op1 = op.pop();
                Character op2 = Optional.ofNullable(op.peek()).orElse('+');
                if(op2=='-'){
                    if(op1=='+'){
                        num.push(n2-n1);
                    }else{
                        num.push(n1+n2);
                    }
                }else{
                    if(op1=='+'){
                        num.push(n1+n2);
                    }else{
                        num.push(n2-n1);
                    }
                }
            }
            return num.peek();
        }
    }


}
