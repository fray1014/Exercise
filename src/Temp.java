import java.util.*;

public class Temp {
    public static void main(String[] args){

        System.out.print(kmpSearch("abcabababa","aba"));
    }
    public static int knapSack(int[] weight, int[] value, int V){
        int n=weight.length;
        int[] dp=new int[V+1];
        for(int i=0;i<n;i++){
            for(int j=V;j>=weight[i];j--){
                dp[j]=Math.max(dp[j-weight[i]]+value[i],dp[j]);
            }
        }
        return dp[V];
    }
    public static int[] prefixTable(String pattern,int n){
        int[] prefix=new int[n];
        int len=0;
        int i=1;
        while(i<n){
            if(pattern.charAt(i)==pattern.charAt(len)){
                prefix[i++]=++len;
            }else{
                if(len>0){
                    len=prefix[len-1];
                }else{
                    prefix[i++]=len;
                }
            }
        }
        return prefix;
    }
    public static void movePrefixTable(int[] prefix,int n){
        for(int i=n-1;i>0;i--){
            prefix[i]=prefix[i-1];
        }
        prefix[0]=-1;
    }
    public static int kmpSearch(String text,String pattern){
        int n=pattern.length();
        int cnt=0;
        int[] prefix=prefixTable(pattern,n);
        movePrefixTable(prefix,n);
        int i=0;
        int j=0;
        while(i<text.length()){
            if(j==n-1&&text.charAt(i)==pattern.charAt(j)){
                cnt++;
                j=prefix[j];
            }
            if(text.charAt(i)==pattern.charAt(j)){
                i++;
                j++;
            }else{
                j=prefix[j];
                if(j==-1){
                    i++;
                    j++;
                }
            }
        }
        return cnt;
    }
}
