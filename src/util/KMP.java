package util;

public class KMP{
    public int[] prefixTable(String pattern,int n){
        int[] prefix=new int[n];
        prefix[0]=0;
        int len=0;
        int i=1;
        while(i<n){
            if(pattern.charAt(i)==pattern.charAt(len)){
                prefix[i++]=++len;
            }else{
                //要再往前检查，又因为prefix[len]是由prefix[len-1]得到的
                if(len>0)
                    len=prefix[len-1];
                else{
                    prefix[i++]=len;
                }
            }
        }
        return prefix;
    }
    public void movePrefixTable(int[] prefix,int n){
        for(int i=n-1;i>0;i--){
            prefix[i]=prefix[i-1];
        }
        prefix[0]=-1;
    }
    //返回匹配的次数
    public int kmpSearch(String text,String pattern){
        int cnt=0;
        int m=text.length();
        int n=pattern.length();
        int[] prefix=prefixTable(pattern,n);
        movePrefixTable(prefix,n);
        int i=0,j=0;
        while(i<m){
            if(j==n-1&&text.charAt(i)==pattern.charAt(j)){
                //System.out.println(i-n-1);
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

