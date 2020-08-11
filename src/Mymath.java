import java.util.*;
public class Mymath {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        String str = scanner.nextLine();
        //double result1 = compute("-3*-3");
        System.out.println(calc(str));
    }
/*
    public static int priority(char s) {
        switch (s) {
            case '(':
            case ')':
                return 0;
            case '-':
            case '+':
                return 1;
            case '*':
            case '%':
            case '/':
                return 2;
            default:
                return -1;

        }
    }

    public static double compute(double num1, double num2, char s) {
        switch (s) {
            case '(':
            case ')':
                return 0;
            case '-':
                return num1 - num2;
            case '+':
                return num1 + num2;
            case '%':
                return num1 % num2;
            case '*':
                return num1 * num2;
            case '/':
                return num1 / num2;
            default:
                return 0;

        }
    }

    public static double compute(String str) {
        double num[] = new double[20];
        int flag = 0, begin = 0, end = 0, now;
        now = -1;
        Stack<Character> st = new Stack<Character>();
        for (int i = 0; i < str.length(); i++) {
            char s = str.charAt(i);
            if (s == ' ') {

            } else if (s == '+' || s == '-' || s == '*' || s == '/' || s == '(' || s == ')' || s == '%') {
                if (flag == 1) {
                    now += 1;
                    if (end < begin) {
                        num[now] = Integer.valueOf(str.substring(begin, begin + 1));
                    } else {
                        num[now] = Integer.valueOf(str.substring(begin, end + 1));
                    }
                    // System.out.println(num[now]);
                    flag = 0;
                }
                if (s == '-') {
                    if (i == 0) {
                        flag = 1;
                        begin = 0;
                    } else if (str.charAt(i - 1) == '(' || str.charAt(i - 1) == '*'
                            || str.charAt(i - 1) == '/') {
                        flag = 1;
                        begin = i;
                    }
                    else {
                        if (st.empty()) {
                            st.push(s);
                        } else if (s == ')') {
                            num[now - 1] = compute(num[now - 1], num[now], st.pop());
                            now -= 1;
                            st.pop();
                        } else if (s == '(') {
                            st.push(s);
                        } else if (priority(s) <= priority(st.peek())) {
                            num[now - 1] = compute(num[now - 1], num[now], st.pop());
                            now -= 1;
                            st.push(s);
                        } else {
                            st.push(s);
                        }
                    }
                } else if (st.empty()) {
                    st.push(s);
                } else if (s == ')') {
                    num[now - 1] = compute(num[now - 1], num[now], st.pop());
                    now -= 1;
                    st.pop();
                } else if (s == '(') {
                    st.push(s);
                } else if (priority(s) <= priority(st.peek())) {
                    num[now - 1] = compute(num[now - 1], num[now], st.pop());
                    now -= 1;
                    st.push(s);
                } else {
                    st.push(s);
                }

            } else if (flag == 0) {
                flag = 1;
                begin = i;
            } else {
                end = i;
            }

        }
        if (flag == 1) {
            now += 1;
            if (end < begin) {
                num[now] = Integer.valueOf(str.substring(begin, begin + 1));
            } else {
                num[now] = Integer.valueOf(str.substring(begin, end + 1));
            }
            // System.out.println(num[now]);
        }
        while (now > 0) {
            num[now - 1] = compute(num[now - 1], num[now], st.pop());
            now -= 1;
        }
        return num[0];
    }*/

    public static String calc(String stat){
        char[] c=stat.toCharArray();
        if(c.length==0)
            return "bad param";
        Stack<String> num=new Stack<>();
        Stack<Character> op=new Stack<>();
        int start=1;
        if(c[0]=='-'){
            num.push("0");
            op.push('-');
        }else if(c[0]-'0'>=0&&c[0]-'0'<=9){
            num.push(Character.toString(c[0]));
        }else if(c[0]=='('){
            String str=calcK(c);
            num.push(calc(str));
            start+=str.length()+1;
        }else{
            return "bad param";
        }
        for(int i=start;i<c.length;i++){
            //当前符号为运算符
            if(num.size()>op.size()){
                //优先级小于等于op栈顶，则弹出计算
                if(op.isEmpty())
                    op.push(c[i]);
                else if(pri(c[i])<=pri(op.peek())){
                    int num2=Integer.parseInt(num.pop());
                    int num1=Integer.parseInt(num.pop());
                    char opc=op.pop();
                    num.push(String.valueOf(calc(num1,num2,opc)));
                    op.push(c[i]);
                }
                else{
                    op.push(c[i]);
                }
            }else{
                //判断括号
                if(c[i]=='('){
                    char[] k=Arrays.copyOfRange(c,i,c.length);
                    String str=calcK(k);
                    i+=str.length()+1;
                    num.push(calc(str));
                }else
                    num.push(Character.toString(c[i]));
            }
        }
        if(num.size()<=op.size())
            return "bad param";

        while(!op.isEmpty()){
            int num2=Integer.parseInt(num.pop());
            int num1=Integer.parseInt(num.pop());
            char opc=op.pop();
            if(!op.isEmpty()){
                char nextop=op.pop();
                if(nextop=='-')
                    num1=-num1;
                op.push('+');
            }
            num.push(String.valueOf(calc(num1,num2,opc)));
        }
        return num.peek();
    }

    public static int pri(char op){
        if(op=='+'||op=='-')
            return 1;
        else if(op=='*'||op=='/')
            return 2;
        return 3;
    }

    public static int calc(int num1,int num2,char opc){
        switch (opc) {
            case '-':
                return num1 - num2;
            case '+':
                return num1 + num2;
            case '*':
                return num1 * num2;
            case '/':
                return num1 / num2;
            default:
                return 0;

        }
    }

    //kuohaoti
    public static String calcK(char[] cin){
        int lcnt=0;
        int rcnt=0;
        int end=0;
        for(int i=0;i<cin.length;i++){
            if(cin[i]=='(')
                lcnt++;
            else if(cin[i]==')')
                rcnt++;
            if(lcnt==rcnt){
                end=i;
                break;
            }
        }
        char[] cout=Arrays.copyOfRange(cin,1,end);
        return String.valueOf(cout);
    }
}
