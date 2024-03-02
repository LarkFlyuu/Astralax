
#include <iostream>

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        if (left == right) return head;

        // get reverse list
        int listSize = right - left + 1;
        ListNode* leftNode = nullptr;
        ListNode* rightNode = nullptr;
        ListNode* list[listSize];
        ListNode* iterNode = head;
        for (int i = 1; i <= right + 1; i++) {
            if (i < left) {
                leftNode = iterNode;
            } else if (i == right + 1) {
                rightNode = iterNode;
                break;
            } else {
                list[i - left] = iterNode;
            }
            iterNode = iterNode->next;
        }

        // reverse list
        for (int i = listSize - 1; i >0; i--) {
            list[i]->next = list[i - 1];
        }

        // connect list
        list[0]->next = rightNode;
        if (leftNode == nullptr) {
            return list[listSize - 1];
        }
        leftNode->next = list[listSize - 1];
        return head;
    }
};

int main() {
    ListNode* head = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))));
    Solution solution;
    solution.reverseBetween(head, 2, 4);
    while (head != nullptr) {
        std::cout << head->val << " ";
        head = head->next;
    }
    std::cout << "\n";
    return 0;
}