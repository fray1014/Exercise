import org.junit.Test;
import util.ListNode;
import util.TreeNode;

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
}
