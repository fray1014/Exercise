import java.util.*;
import util.*;
import org.junit.Test;
public class BinaryTree {
    @Test
    public void test(){
        TreeNode t0=new TreeNode(0);
        TreeNode t1=new TreeNode(1);
        TreeNode t2=new TreeNode(2);
        TreeNode t3=new TreeNode(3);
        TreeNode t4=new TreeNode(4);
        TreeNode t5=new TreeNode(5);
        TreeNode t6=new TreeNode(6);
        TreeNode t7=new TreeNode(7);
        t0.left=t1;
        t0.right=t2;
        t1.left=t3;
        t1.right=t4;
        t2.left=t5;
        t2.right=t6;
        t4.left=t7;
        levelOrderTraverse(t0);
    }
    /**递归前序*/
    public void preOrderTraverse1(TreeNode root){
        if(root!=null){
            System.out.println(root.val);
            preOrderTraverse1(root.left);
            preOrderTraverse1(root.right);
        }
    }
    /**非递归前序*/
    public void preOrderTraverse2(TreeNode root){
        Stack<TreeNode> stack=new Stack<>();
        while(root!=null||!stack.isEmpty()){
            if(root!=null){
                System.out.println(root.val);
                stack.push(root);
                root=root.left;
            }else{
                root=stack.pop().right;
            }
        }
    }
    /**递归中序*/
    public void inOrderTraverse1(TreeNode root){
        if(root!=null){
            inOrderTraverse1(root.left);
            System.out.println(root.val);
            inOrderTraverse1(root.right);
        }
    }
    /**非递归中序*/
    public void inOrderTraverse2(TreeNode root){
        Stack<TreeNode> stack=new Stack<>();
        while(root!=null||!stack.isEmpty()){
            if(root!=null){
                stack.push(root);
                root=root.left;
            }else{
                System.out.println(stack.peek().val);
                root=stack.pop().right;
            }
        }
    }
    /**递归后序*/
    public void postOrderTraverse1(TreeNode root){
        if(root!=null){
            postOrderTraverse1(root.left);
            postOrderTraverse1(root.right);
            System.out.println(root.val);
        }
    }
    /**非递归后序*/
    public void postOrderTraverse2(TreeNode root){
        TreeNode cur, pre = null;

        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);

        while (!stack.empty()) {
            cur = stack.peek();
            if ((cur.left == null && cur.right == null) || (pre != null && (pre == cur.left || pre == cur.right))) {
                System.out.println(cur.val);
                stack.pop();
                pre = cur;
            } else {
                if (cur.right != null)
                    stack.push(cur.right);
                if (cur.left != null)
                    stack.push(cur.left);
            }
        }
    }
    /**层次遍历*/
    public void levelOrderTraverse(TreeNode root){
        Queue<TreeNode> queue=new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode cur=queue.poll();
            System.out.println(cur.val);
            if(cur.left!=null)
                queue.offer(cur.left);
            if(cur.right!=null)
                queue.offer(cur.right);
        }
    }
    /**给中序后序，还原二叉树*/
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        if(inorder.length==0||postorder.length==0)
            return null;
        int rootidx=postorder.length-1;
        TreeNode root=new TreeNode(postorder[rootidx]);
        for(int i=0;i<inorder.length;i++){
            if(inorder[i]==postorder[rootidx]){
                int[] ileft= Arrays.copyOfRange(inorder,0,i);
                int[] iright=Arrays.copyOfRange(inorder,i+1,inorder.length);
                int[] pleft=Arrays.copyOfRange(postorder,0,i);
                int[] pright=Arrays.copyOfRange(postorder,i,inorder.length-1);
                root.left=buildTree(ileft,pleft);
                root.right=buildTree(iright,pright);
            }
        }
        return root;
    }
    /**给前序中序，还原二叉树*/
    public TreeNode buildTree2(int[] preorder,int[] inorder){
        if(preorder.length==0||inorder.length==0)
            return null;
        TreeNode root=new TreeNode(preorder[0]);
        for(int i=0;i<inorder.length;i++){
            if(inorder[i]==preorder[0]){
                int[] ileft= Arrays.copyOfRange(inorder,0,i);
                int[] iright=Arrays.copyOfRange(inorder,i+1,inorder.length);
                int[] pleft=Arrays.copyOfRange(preorder,1,i+1);
                int[] pright=Arrays.copyOfRange(preorder,i+1,preorder.length);
                root.left=buildTree2(pleft,ileft);
                root.right=buildTree2(pright,iright);
            }
        }
        return root;
    }
    /**给前序后序，还原二叉树*/
    public TreeNode buildTree3(int[] preorder,int[] postorder){
        if(preorder.length==0||postorder.length==0)
            return null;
        TreeNode root=new TreeNode(preorder[0]);
        if(preorder.length>1){
            for(int i=0;i<postorder.length;i++){
                if(postorder[i]==preorder[1]){
                    int[] preleft=Arrays.copyOfRange(preorder,1,i+2);
                    int[] preright=Arrays.copyOfRange(preorder,i+2,preorder.length);
                    int[] postleft=Arrays.copyOfRange(postorder,0,i+1);
                    int[] postright=Arrays.copyOfRange(postorder,i+1,postorder.length-1);
                    root.left=buildTree3(preleft,postleft);
                    root.right=buildTree3(preright,postright);
                }
            }
        }
        return root;
    }
    @Test
    public void test2(){
        int[] pre={1,2,4,7,5,3,6};
        int[] in={7,4,2,5,1,3,6};
        int[] post={7,4,5,2,6,3,1};
        int[] t0={1,2};
        int[] t1={2,1};
        TreeNode tmp=buildTree3(t0,t1);
        System.out.print(tmp.val);
    }
}
