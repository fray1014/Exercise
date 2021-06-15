import org.junit.Test;
import util.ListNode;
import util.TreeNode;

import java.util.*;

public class LeetCode5 {
    @Test
    public void test(){
        Solution78 s = new Solution78();
        Integer a = 1;
        Integer b = 2;
        Integer c = 3;
        Integer d = 3;
        Integer e = 321;
        Integer f = 321;
        Long g = 3L;
//        System.out.println(c == d);
//        System.out.println(e == f);
//        System.out.println(c == (a + b));
//        System.out.println(c.equals(a + b));
        System.out.println(g == (a + b));
//        System.out.println(g.equals(a + b));
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

    class Solution70 {
        public int climbStairs(int n) {
            int num1 = 0;
            int num2 = 1;
            int cur = 0;
            for(int i = 1;i<=n;i++){
                cur = num1 + num2;
                num1 = num2;
                num2 = cur;
            }
            return cur;
        }
    }

    class Solution0404 {
        public boolean isBalanced(TreeNode root) {
            if(root==null){
                return true;
            }else{
                return Math.abs(level(root.left) - level(root.right)) <= 1
                        &&isBalanced(root.left)&&isBalanced(root.right);
            }
        }

        public int level(TreeNode root){
            if(root==null){
                return 0;
            }
            return Math.max(level(root.left),level(root.right))+1;
        }
    }

    class Solution1254 {
        public int closedIsland(int[][] grid) {
            if(grid==null){
                return 0;
            }
            int row = grid.length;
            int col = grid[0].length;
            int cnt = 0;
            for(int i=1;i<row;i++){
                for(int j=1;j<col;j++){
                    if(grid[i][j]==0){
                        cnt++;
                        dfs(grid,i,j,row,col);
                    }
                }
            }
            return cnt;
        }

        public void dfs(int[][] grid,int x,int y,int row,int col){
            if(x<=0||x>=row-1||y<=0||y>=col-1||grid[x][y]==1){
                return;
            }
            grid[x][y] = 1;
            dfs(grid,x-1,y,row,col);
            dfs(grid,x-1,y,row,col);
            dfs(grid,x,y-1,row,col);
            dfs(grid,x,y+1,row,col);
        }
    }
    class Solution1482 {
        public int minDays(int[] bloomDay, int m, int k) {
            //二分查找
            if(m*k>bloomDay.length){
                return -1;
            }
            int low = Integer.MAX_VALUE;
            int high = 0;
            for(int n : bloomDay){
                low = Math.min(low,n);
                high = Math.max(high,n);
            }
            while (low < high){
                int days = (high - low)/2 + low;
                if(canMake(bloomDay,m,k,days)){
                    high = days;
                }else{
                    low = days + 1;
                }
            }
            return low;
        }

        public boolean canMake(int[] bloomDay, int m, int k, int days){
            int cnt = 0;
            int flowers = 0;
            for(int i = 0; i < bloomDay.length && cnt < m; i++){
                if(bloomDay[i]<=days){
                    flowers++;
                    if(flowers==k){
                        cnt++;
                        flowers=0;
                    }
                }else{
                    flowers=0;
                }
            }
            return cnt>=m;
        }
    }

    class Solution78 {
        List<Integer> t = new ArrayList<Integer>();
        List<List<Integer>> ans = new ArrayList<List<Integer>>();

        public List<List<Integer>> subsets(int[] nums) {
            dfs(0, nums);
            return ans;
        }

        public void dfs(int cur, int[] nums) {
            if (cur == nums.length) {
                ans.add(new ArrayList<Integer>(t));
                for(int tmp:t){
                    System.out.print(tmp);
                }
                System.out.println();
                return;
            }
            t.add(nums[cur]);
            dfs(cur + 1, nums);
            t.remove(t.size() - 1);
            dfs(cur + 1, nums);

        }
    }

}
