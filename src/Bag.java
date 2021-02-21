import org.junit.Test;

public class Bag {
    public static int knapSack(int[] weight, int[] value, int V){
        int N=weight.length;
        int[] dp = new int[V+1];
        for(int i=1;i<N+1;i++){
            //逆序实现
            for(int j=V;j>=weight[i-1];j--){
                dp[j] = Math.max(dp[j-weight[i-1]]+value[i-1],dp[j]);
            }
        }
        return dp[V];
    }

    @Test
    public void test(){
        int V=8;
        int[] weight={3,5,1,2,2,4,1};
        int[] value={4,5,2,1,3,4,3};
        System.out.println(knapSack(weight,value,V));
    }
}
