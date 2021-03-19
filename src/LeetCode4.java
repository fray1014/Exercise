import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.Test;
import util.*;
public class LeetCode4 {
    /**翻转二叉树以匹配前序遍历*/
    class Solution971 {
        int depth=0;
        public List<Integer> flipMatchVoyage(TreeNode root, int[] voyage) {
            List<Integer> res=new ArrayList<>();
            if(root==null||voyage.length==0||voyage==null)
                return res;
            dfs(root,voyage,res);
            if(!res.isEmpty()&&res.get(0)==-1){
                res.clear();
                res.add(-1);
            }
            return res;
        }
        //是否需要翻转
        public void dfs(TreeNode root,int[] voyage,List<Integer> res){
            if(root!=null){
                if(root.val!=voyage[depth++]){
                    res.clear();
                    res.add(-1);
                    return;
                }
                if(depth<voyage.length&&root.left!=null&&root.left.val!=voyage[depth]){
                    res.add(root.val);
                    dfs(root.right,voyage,res);
                    dfs(root.left,voyage,res);
                }else{
                    dfs(root.left,voyage,res);
                    dfs(root.right,voyage,res);
                }
            }
        }
    }
    /**左叶子之和*/
    class Solution404 {
        public int sumOfLeftLeaves(TreeNode root) {
            return root==null?0:dfs(root);
        }
        public int dfs(TreeNode root){
            int sum=0;
            if(root.left!=null){
                if(isLeaf(root.left)){
                    sum+=root.left.val;
                }else{
                    sum+=dfs(root.left);
                }
            }
            if(root.right!=null&&!isLeaf(root.right)){
                sum+=dfs(root.right);
            }
            return sum;
        }
        public boolean isLeaf(TreeNode root){
            return root.left==null&&root.right==null;
        }
    }
    /**子集*/
    class Solution78 {
        //dfs+回溯
        public List<List<Integer>> subsets(int[] nums) {
            List<List<Integer>> res=new LinkedList<>();
            if(nums.length==0||nums==null)
                return res;
            boolean[] used=new boolean[nums.length];
            Deque<Integer> path=new LinkedList<>();
            Arrays.sort(nums);
            for(int i=0;i<=nums.length;i++){
                dfs(nums,used,i,0,path,res);
            }
            return res;
        }
        public void dfs(int[] nums,boolean[] used,int len,int begin,Deque<Integer> path,List<List<Integer>> res){
            if(path.size()==len){
                res.add(new LinkedList<>(path));
                return;
            }
            for(int i=begin;i<nums.length;i++){
                if(!used[i]){
                    path.addLast(nums[i]);
                    used[i]=true;
                    dfs(nums,used,len,i+1,path,res);
                    used[i]=false;
                    path.removeLast();
                }
            }
        }
        //二进制枚举，妙
        public List<List<Integer>> subsets2(int[] nums){
            List<List<Integer>> res=new LinkedList<>();
            for(int i=0;i<(1<<nums.length);i++){
                List<Integer> tmp=new LinkedList<>();
                for(int j=0;j<nums.length;j++){
                    if((i&(1<<j))!=0){
                        tmp.add(nums[j]);
                    }
                }
                res.add(tmp);
            }
            return res;
        }
    }

    /**累加树*/
    class Solution538 {
        int sum=0;
        public TreeNode convertBST(TreeNode root) {
            if(root!=null){
                convertBST(root.right);
                sum+=root.val;
                root.val=sum;
                convertBST(root.left);
            }
            return root;
        }
    }
    /**用01拼字符串*/
    class Solution474 {
        public int findMaxForm(String[] strs, int m, int n) {
            //m个0，n个1
            int[][] dp=new int[m+1][n+1];
            for(String s:strs){
                int[] c=count(s);
                for(int i=m;i>=c[0];i--){
                    for(int j=n;j>=c[1];j--){
                        dp[i][j]=Math.max(dp[i][j],dp[i-c[0]][j-c[1]]+1);
                    }
                }
            }
            return dp[m][n];
        }
        public int[] count(String str){
            int[] ret=new int[2];
            for(int i=0;i<str.length();i++){
                ret[str.charAt(i)-'0']++;
            }
            return ret;
        }
    }
    /**合并二叉树*/
    class Solution617 {
        public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
            TreeNode res;
            if(t1==null&&t2==null)
                return null;
            res=new TreeNode((t1==null?0:t1.val)+(t2==null?0:t2.val));
            res.left=mergeTrees(t1==null?null:t1.left,t2==null?null:t2.left);
            res.right=mergeTrees(t1==null?null:t1.right,t2==null?null:t2.right);
            return res;
        }
    }
    /**二叉树最大宽度*/
    class Solution662 {
        Queue<TreeNode> q0 = new LinkedList<>();
        Queue<TreeNode> q1 = new LinkedList<>();
        Deque<Integer> dq0 = new ArrayDeque<>();
        Deque<Integer> dq1 = new ArrayDeque<>();
        int maxwidth = 0;

        public int widthOfBinaryTree(TreeNode root) {
            if (root == null)
                return 0;
            if(root.left==null&&root.right==null)
                return 1;
            q0.offer(root);
            dq0.addLast(1);
            while (!q0.isEmpty()) {
                TreeNode tmp = q0.poll();
                int index = dq0.removeFirst();
                if (tmp.left != null) {
                    q1.offer(tmp.left);
                    dq1.addLast(2 * index - 1);
                }
                if (tmp.right != null) {
                    q1.offer(tmp.right);
                    dq1.addLast(2 * index);
                }
                //这一层遍历完了
                if (q0.isEmpty()) {
                    q0 = new LinkedList<>(q1);
                    dq0 = new ArrayDeque<>(dq1);
                    //结束了
                    if (dq1.isEmpty())
                        break;
                    maxwidth = Math.max(maxwidth, dq1.getLast() - dq1.getFirst() + 1);
                    q1.clear();
                    dq1.clear();
                }
            }
            return maxwidth;
        }
    }
    /**数组中出现一次的数字*/
    class Solution136{
        //异或运算
        public int singleNumber(int[] nums) {
            int res=0;
            for(int i:nums){
                res^=i;
            }
            return res;
        }
    }
    /**只有两个键的键盘*/
    class Solution650{
        //质因数相加
        public int minSteps(int n) {
            int res=0;
            int base=2;
            while(n>1){
                if(n%base==0){
                    res+=base;
                    n/=base;
                }else{
                    base++;
                }
            }
            return res;
        }
    }
    /**从根节点到叶节点的路径总和*/
    class Solution113 {
        List<List<Integer>> res=new LinkedList<>();
        List<Integer> path=new LinkedList<>();
        public List<List<Integer>> pathSum(TreeNode root, int sum) {
            dfs(root,sum);
            return res;
        }
        public void dfs(TreeNode root,int sum){
            if(root==null)
                return;
            path.add(root.val);
            sum-=root.val;
            if(root.left==null&&root.right==null&&sum==0){
                res.add(new LinkedList<>(path));
            }
            dfs(root.left,sum);
            dfs(root.right,sum);
            path.remove(path.size()-1);
        }
    }
    /**公共祖先结点*/
    class Solution235 {
        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            while(true){
                if(root.val<p.val&&root.val<q.val)
                    root=root.right;
                else if(root.val>q.val&&root.val>p.val)
                    root=root.left;
                else
                    break;
            }
            return root;
        }
    }
    /**填充二叉树结点右侧的next指针*/
    class Solution117 {
        Node last = null, nextStart = null;
        public Node connect(Node root) {
            if(root==null)
                return null;
            Node start=root;
            while(start!=null){
                last=null;
                nextStart=null;
                for(Node p=start;p!=null;p=p.next){
                    if (p.left != null) {
                        handle(p.left);
                    }
                    if (p.right != null) {
                        handle(p.right);
                    }
                }
                start=nextStart;
            }
            return root;
        }
        public void handle(Node p) {
            if (last != null) {
                last.next = p;
            }
            if (nextStart == null) {
                nextStart = p;
            }
            last = p;
        }
    }
    /**四数之和*/
    class Solution18{
        public List<List<Integer>> fourSum(int[] nums, int target) {
            List<List<Integer>> res=new LinkedList<>();
            List<Integer> path=new LinkedList<>();
            if(nums==null||nums.length==0)
                return res;
            Arrays.sort(nums);
            boolean[] used=new boolean[nums.length];
            dfs(nums,target,used,0,0,res,path);
            return res;
        }
        public void dfs(int[] nums,int target,boolean[] used,int depth,int begin,List<List<Integer>> res,List<Integer> path){
            if(target==0&&depth==4){
                res.add(new LinkedList<>(path));
                return;
            }
            if(begin>=nums.length||depth==4)
                return;
            if(nums[begin]*(4-depth)>target)
                return;
            if(nums[nums.length-1]*(4-depth)<target)
                return;
            for(int i=begin;i<nums.length;i++){
                if(!used[i]){
                    if(i>0&&used[i-1]==false&&nums[i-1]==nums[i])
                        continue;
                    used[i]=true;
                    path.add(nums[i]);
                    dfs(nums,target-nums[i],used,depth+1,i+1,res,path);
                    path.remove(path.size()-1);
                    used[i]=false;
                }
            }
        }
    }
    /**两数相加*/
    class Solution2 {
        public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
            ListNode res=new ListNode();
            ListNode head=res;
            int c=0;
            while(l1!=null&&l2!=null){
                int newval=l1.val+l2.val;
                ListNode tmp=new ListNode((newval+c)%10);
                c=(newval+c)/10;
                res.next=tmp;
                res=res.next;
                l1=l1.next;
                l2=l2.next;
            }
            if(l1==null&&l2!=null){
                while(l2!=null){
                    ListNode tmp=new ListNode((l2.val+c)%10);
                    res.next=tmp;
                    res=res.next;
                    c=(l2.val+c)/10;
                    l2=l2.next;
                }
            }
            if(l2==null&&l1!=null){
                while(l1!=null){
                    ListNode tmp=new ListNode((l1.val+c)%10);
                    res.next=tmp;
                    res=res.next;
                    c=(l1.val+c)/10;
                    l1=l1.next;
                }
            }
            if(c==1)
                res.next=new ListNode(1);
            return head.next==null?null:head.next;
        }
    }
    /**二叉树展开为链表*/
    class Solution114 {
        public void flatten(TreeNode root) {
            List<TreeNode> l=new ArrayList<>();
            if(root==null)
                return;
            dfs(root,l);
            for(int i=1;i<l.size();i++){
                TreeNode prev=l.get(i-1);
                TreeNode cur=l.get(i);
                prev.left=null;
                prev.right=cur;
            }
        }
        public void dfs(TreeNode root,List<TreeNode> l){
            if(root!=null){
                l.add(root);
                dfs(root.left,l);
                dfs(root.right,l);
            }
        }
        public void flatten2(TreeNode root) {
            TreeNode curr = root;
            while (curr != null) {
                if(curr.left!=null){
                    TreeNode next=curr.left;
                    TreeNode predecessor=next;
                    while(predecessor.right!=null){
                        predecessor=predecessor.right;
                    }
                    predecessor.right=curr.right;
                    curr.right=next;
                    curr.left=null;
                }
                curr=curr.right;
            }
        }
    }
    /**查找常用字符串*/
    class Solution1002 {
        public List<String> commonChars(String[] A) {
            int[] minfre=new int[26];
            for(int j=0;j<A.length;j++){
                int[] freq=new int[26];
                for(int i=0;i<A[j].length();i++){
                    freq[A[j].charAt(i)-'a']++;
                }
                if(j==0)
                    minfre=freq;
                else{
                    for(int i=0;i<26;i++){
                        minfre[i]=Math.min(minfre[i],freq[i]);
                    }
                }
            }
            List<String> res=new LinkedList<>();
            for(int i=0;i<26;i++){
                while(minfre[i]!=0){
                    minfre[i]--;
                    res.add(Character.toString('a'+i));
                }
            }
            return res;
        }
    }
    /**长按键入*/
    class Solution925 {
        public boolean isLongPressedName(String name, String typed) {
            /*正则表达式
            StringBuffer pattern=new StringBuffer();
            for(int i=0;i<name.length();i++){
                pattern.append(name.charAt(i)+"+");
            }
            return Pattern.matches(pattern.toString(),typed);*/
            int i = 0, j = 0;
            while (j < typed.length()) {
                if (i < name.length() && name.charAt(i) == typed.charAt(j)) {
                    i++;
                    j++;
                } else if (j > 0 && typed.charAt(j) == typed.charAt(j - 1)) {
                    j++;
                } else {
                    return false;
                }
            }
            return i == name.length();
        }
    }
    /**重排链表*/
    class Solution143 {
        public void reorderList(ListNode head) {
            //最后一个点是n/2，先得到一个逆序的
            int index=0;
            int len=0;
            ListNode tmp=head;
            while(tmp!=null){
                len++;
                tmp=tmp.next;
            }
            tmp=head;
            while(index<len/2){
                index++;
                tmp=tmp.next;
            }
            ListNode pre=null;
            while(tmp!=null){
                ListNode next=tmp.next;
                tmp.next=pre;
                pre=tmp;
                tmp=next;
            }
            //pre为逆序链表头，现在开始互插
            while(pre!=null){
                ListNode pnext=pre.next;
                ListNode hnext;
                if(pnext==null)
                    hnext=null;
                else
                    hnext=head.next;
                head.next=pre;
                pre.next=hnext;
                head=hnext;
                pre=pnext;
            }
        }
    }
    /**有序数组的平方*/
    class Solution977 {
        public int[] sortedSquares(int[] A) {
            if(A==null||A.length==0)
                return A;
            int[] res=new int[A.length];
            int st=0;
            int ed=A.length-1;
            int index=ed;
            while(st<=ed){
                if(A[st]*A[st]>A[ed]*A[ed]){
                    res[index--]=A[st]*A[st];
                    st++;
                }else{
                    res[index--]=A[ed]*A[ed];
                    ed--;
                }
            }
            return res;
        }
    }
    /**课程表*/
    class Solution207 {
        public boolean canFinish(int numCourses, int[][] prerequisites) {
            List<List<Integer>> adj=new ArrayList<>();
            for(int i=0;i<numCourses;i++)
                adj.add(new ArrayList<>());
            int[] flags=new int[numCourses];
            for(int[] cp:prerequisites)
                adj.get(cp[1]).add(cp[0]);
            for(int i=0;i<numCourses;i++){
                if(!dfs(adj,flags,i))
                    return false;
            }
            return true;
        }
        private boolean dfs(List<List<Integer>> adjacency, int[] flags, int i) {
            if(flags[i] == 1) return false;
            if(flags[i] == -1) return true;
            flags[i] = 1;
            for(Integer j : adjacency.get(i))
                if(!dfs(adjacency, flags, j)) return false;
            flags[i] = -1;
            return true;
        }
    }
    /**划分字母区间*/
    class Solution763 {
        //哈希法
        public List<Integer> partitionLabels(String S) {
            List<Integer> res=new LinkedList<>();
            if(S==null||S.length()==0)
                return res;
            HashSet<Character> hs=new HashSet<>();
            int len;
            int j;
            for(int i=0;i<S.length();i++){
                len=1;
                j=findLast(S,S.charAt(i));
                int st=i;
                int ed=j;
                while(st<ed){
                    len++;
                    st++;
                    if(hs.contains(S.charAt(st)))
                        continue;
                    hs.add(S.charAt(st));
                    ed=Math.max(findLast(S,S.charAt(st)),ed);
                }
                hs.clear();
                i=ed;
                if(len!=0)
                    res.add(len);
            }
            return res;
        }
        //数组记录字母最后出现的下标
        public List<Integer> partitionLabels2(String S) {
            int[] last = new int[26];
            int length = S.length();
            for (int i = 0; i < length; i++) {
                last[S.charAt(i) - 'a'] = i;
            }
            List<Integer> partition = new ArrayList<Integer>();
            int start = 0, end = 0;
            for (int i = 0; i < length; i++) {
                end = Math.max(end, last[S.charAt(i) - 'a']);
                if (i == end) {
                    partition.add(end - start + 1);
                    start = end + 1;
                }
            }
            return partition;
        }
        public int findLast(String S,char c){
            int index=S.length()-1;
            while(S.charAt(index)!=c){
                index--;
            }
            return index;
        }
    }
    /**最长上升子串*/
    class Solution300 {
        public int lengthOfLIS(int[] nums) {
            int len = 1, n = nums.length;
            if (n == 0) {
                return 0;
            }
            int[] d = new int[n + 1];
            d[len] = nums[0];
            for (int i = 1; i < n; ++i) {
                if (nums[i] > d[len]) {
                    d[++len] = nums[i];
                } else {
                    int l = 1, r = len, pos = 0; // 如果找不到说明所有的数都比 nums[i] 大，此时要更新 d[1]，所以这里将 pos 设为 0
                    while (l <= r) {
                        int mid = (l + r) >> 1;
                        if (d[mid] < nums[i]) {
                            pos = mid;
                            l = mid + 1;
                        } else {
                            r = mid - 1;
                        }
                    }
                    d[pos + 1] = nums[i];
                }
            }
            return len;
        }
    }

    class Solution1438{
        public int longestSubarray(int[] nums, int limit) {
            //递增
            Deque<Integer> dqMin=new LinkedList<>();
            //递减
            Deque<Integer> dqMax=new LinkedList<>();
            int left=0,right=0,ret=0;
            while(right<nums.length){
                while(!dqMax.isEmpty()&&nums[right]>dqMax.peekLast()){
                    dqMax.pollLast();
                }
                while(!dqMin.isEmpty()&&nums[right]<dqMin.peekLast()){
                    dqMin.pollLast();
                }
                dqMax.offerLast(nums[right]);
                dqMin.offerLast(nums[right]);
                while(!dqMax.isEmpty()&&!dqMin.isEmpty()&&dqMax.peekFirst()-dqMin.peekFirst()>limit){
                    if(nums[left]==dqMax.peekFirst()){
                        dqMax.pollFirst();
                    }
                    if(nums[left]==dqMin.peekFirst()){
                        dqMin.pollFirst();
                    }
                    left++;
                }
                ret=Math.max(ret,right-left+1);
                right++;
            }
            return ret;
        }
    }

    class Solution766{
        public boolean isToeplitzMatrix(int[][] matrix) {
            if(matrix==null||matrix.length<=0){
                return false;
            }
            int row = matrix.length;
            int col = matrix[0].length;
            int x = row - 2;
            int y = 0;
            while(x>=0){
                int oldx = x;
                while(y<col-1&&x<row-1){
                    if(matrix[x][y]!=matrix[++x][++y]){
                        return false;
                    }
                }
                x = oldx - 1;
                y = 0;
            }
            x = 0;
            while(y<col-1){
                int oldy = y;
                while(y<col-1&&x<row-1){
                    if(matrix[x][y]!=matrix[++x][++y]){
                        return false;
                    }
                }
                y = oldy + 1;
                x = 0;
            }
            return true;
        }
    }

    class Solution1052 {
        public int maxSatisfied(int[] customers, int[] grumpy, int X) {
            /*
            int left = 0;
            int right = X-1;
            int res = 0;
            int maxC = 0;
            int curC = 0;

            for(int i=0;i<X;i++){
                if(i==customers.length){
                    break;
                }
                if(grumpy[i]==1){
                    maxC+=customers[i];
                }else{
                    res+=customers[i];
                }
            }
            right++;
            curC = maxC;
            while(right<customers.length){
                if(grumpy[right]==1){
                    curC+=customers[right];
                }else{
                    res+=customers[right];
                }
                if(grumpy[left]==1){
                    curC-=customers[left];
                }
                left++;
                right++;
                maxC=Math.max(maxC,curC);
            }
            return res+maxC;*/
            int total = 0;
            int n = customers.length;
            for (int i = 0; i < n; i++) {
                if (grumpy[i] == 0) {
                    total += customers[i];
                }
            }
            int increase = 0;
            for (int i = 0; i < X; i++) {
                increase += customers[i] * grumpy[i];
            }
            int maxIncrease = increase;
            for (int i = X; i < n; i++) {
                increase = increase - customers[i - X] * grumpy[i - X] + customers[i] * grumpy[i];
                maxIncrease = Math.max(maxIncrease, increase);
            }
            return total + maxIncrease;
        }
    }

    class Solution832 {
        public int[][] flipAndInvertImage(int[][] A) {
            int row = A.length;
            int col = A[0].length;
            if(row == 0 || col == 0){
                return A;
            }
            int[][] res=new int[row][col];
            for(int i=0;i<row;i++){
                for(int j=0;j<col;j++){
                    res[i][j] = 1-A[i][col-j-1];
                }
            }
            return res;
        }
    }

    class Solution867 {
        public int[][] transpose(int[][] matrix) {
            int row = matrix.length;
            int col = matrix[0].length;
            int[][] res = new int[col][row];
            for(int i=0;i<col;i++){
                for(int j=0;j<row;j++){
                    res[i][j]=matrix[j][i];
                }
            }
            return res;
        }
    }

    class Solution395 {
        public int longestSubstring(String s, int k) {
            if(s.length()<k){
                return 0;
            }
            HashMap<Character,Integer> hm = new HashMap<>();
            for(char c:s.toCharArray()){
                hm.put(c,hm.getOrDefault(c,0)+1);
            }
            for(char c:hm.keySet()){
                if(hm.get(c)<k){
                    int res = 0;
                    for(String str:s.split(String.valueOf(c))){
                        res = Math.max(longestSubstring(str,k),res);
                    }
                    return res;
                }
            }
            return s.length();
        }
    }

    class Solution896 {
        public boolean isMonotonic(int[] A) {
//            if(A.length<=1){
//                return true;
//            }
//            int d = 0;
//            int index = 1;
//            while(index<A.length && A[index] == A[index - 1]){
//                index++;
//            }
//            if(index<A.length){
//                if(A[index]>A[index-1]){
//                    d = 1;
//                }else{
//                    d = 0;
//                }
//            }
//            index++;
//            while(index<A.length){
//                if(d==1&&A[index]<A[index-1]){
//                    return false;
//                }else if(d==0&&A[index]>A[index-1]){
//                    return false;
//                }
//                index++;
//            }
//            return true;
            boolean inc = true;
            boolean dec = true;
            for(int i=1;i<A.length;i++){
                if(A[i]>A[i-1]){
                    dec = false;
                }else if(A[i]<A[i-1]){
                    inc = false;
                }
                if(!(dec||inc)){
                    return false;
                }
            }
            return true;
        }
    }

    class NumArray {
        int[] sums;

        public NumArray(int[] nums) {
            int n = nums.length;
            sums = new int[n + 1];
            for (int i = 0; i < n; i++) {
                sums[i + 1] = sums[i] + nums[i];
            }
        }

        public int sumRange(int i, int j) {
            return sums[j + 1] - sums[i];
        }
    }

    class Solution1047 {
        public String removeDuplicates(String S) {
            StringBuffer stack = new StringBuffer();
            int top = -1;
            for(int i=0;i<S.length();i++){
                char c = S.charAt(i);
                if(top>=0&&c==stack.charAt(top)){
                    stack.deleteCharAt(top);
                    --top;
                }else{
                    stack.append(c);
                    ++top;
                }
            }
            return stack.toString();
        }
    }
    @Test
    public void test(){
        System.out.println();
        System.out.println((int)'1');
    }
}
