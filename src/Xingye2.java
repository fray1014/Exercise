import java.util.*;
public class Xingye2 {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        while(s.hasNext()){
            System.out.println(solve(s.nextLine()));
        }
    }
    public static String solve(String str){
        if(str.length()%2 == 1){
            return "false";
        }
        //去重
        String str2 = isDouble(str);
        if(str2.equals("false")){
            return str2;
        }
        //回文
        if(!isAva(str2)){
            return "false";
        }
        return str2;
    }
    public static String isDouble(String str){
        StringBuilder res = new StringBuilder();
        for(int i=0;i<str.length();i++){
            //奇数下标必须与前一个相同
            if(i%2==1){
                if(str.charAt(i)!=str.charAt(i-1)){
                    return "false";
                }
            }else{
                res.append(str.charAt(i));
            }
        }
        return res.toString();
    }
    public static boolean isAva(String str){
        for(int i=0;i<str.length()/2;i++){
            if(str.charAt(i)!=str.charAt(str.length()-i-1)){
                return false;
            }
        }
        return true;
    }
}
