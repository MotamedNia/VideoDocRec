//
// Created by hossein on 11/10/18.
//

#ifndef LINKEDLIST_LINKEDLIST_H
#define LINKEDLIST_LINKEDLIST_H

#endif //LINKEDLIST_LINKEDLIST_H
#include <iostream>
#include <stdexcept>

using namespace std;


template <class T>
class node
{
public:
    T data;
    node<T> *link;

    node(){
        link=NULL;
    }
};


template <class T>
class LinkedList{

    friend istream &operator>>(istream , LinkedList);
    friend ostream &operator<<(ostream , LinkedList);
public:

    node<T> *ptr;
public:
    LinkedList(){
        ptr=NULL;

    }

    bool Add(T x);
    bool Del(T x);
    node<T>* Search(T x);
    node<T>* SearchBefor(T x);
    int nodeCount();
    void deleteAll();
    LinkedList<T> duplicate(); // Make a copy from own object
    static LinkedList<T> copy(LinkedList<T>);// Make a copy from every LinkedLists objects
    void merg(LinkedList<T>);
    void print();

    T &operator[](int);
    T  operator[](int)const;
};

template <class T>
bool LinkedList<T>::Add(T x){
    node<T> *temp, *d;

    d=new(node<T>);

    temp = SearchBefor(x);


    if(temp !=0){
        d->link=temp->link;
        temp->link=d;
    }
    else{
        d->link=ptr;
        ptr=d;
    }
    d->data =x;

    return true;
}

template <class T>
bool LinkedList<T>::Del(T x){
    node<T> *temp, *d;

    if(Search(x)==0)
        return false;

    temp=SearchBefor(x);
    if(temp !=0){
        d=(temp->link);
        (temp->link)=(d->link);
        delete (d);
    }
    else{
        d=ptr;
        ptr=(d->link);
        delete (d);
    }
    return true;
}

template <class T>
node<T>*  LinkedList<T>::Search(T x){
    node<T> *temp;
    temp =ptr;
    while(temp !=NULL){
        if(temp->data==x)
            return temp;
        temp=temp->link;
    }
    temp=NULL;
    return temp;
}

template <typename T>
node<T>* LinkedList<T>::SearchBefor(T x){
    node<T> *temp;
    node<T> *d;
    d=nullptr;

    temp=ptr;

    if(temp ==nullptr)
        return 0;
    while(temp!=nullptr){
        if((temp->data) >=x)
            return d;
        d=temp;
        temp=(temp->link);
    }
    return d;

}

template <typename T>
int	LinkedList<T>::nodeCount(){
    int c=0;

    node<T> *temp;
    temp=ptr;

    while(temp !=NULL)
    {
        ++c;
        temp=(temp->link);
    }
    return c;
}

template<typename T>
void LinkedList<T>::deleteAll(){
    while(ptr!=0)
        this->Del(ptr->data);

}

template <typename T>
LinkedList<T> LinkedList<T>::duplicate(){
    LinkedList<T> list;
    node<T> *temp;

    temp=(*this).ptr;
    list.ptr=(*this).ptr;
    while(temp!=0){
        list.Add(temp->data);
        temp=(temp->link);
    }
    return list;
}

template <typename T>
LinkedList<T> LinkedList<T>::copy(LinkedList<T> l){
    LinkedList<T> list;
    node<T> *temp;

    temp=l.ptr;
    list.ptr=l.ptr;
    while(temp!=0){
        list.Add(temp->data);
        temp=(temp->link);
    }
    return list;
}

template <typename T>
void LinkedList<T>::merg(LinkedList<T> list){
    node<T> *temp;
    temp=list.ptr;
    while(temp !=NULL)
        this->Add(temp->data);
}

template <typename T>
void LinkedList<T>::print(){
    node<T>* temp;

    temp=ptr;
    while(temp!=NULL){
        cout<<temp->data<<"	";
        temp=(temp->link);
    }
}

template<typename T>
ostream &operator<<(ostream output, LinkedList<T> l){
    l.print();

    return output;
}

template<typename T>
T &LinkedList<T>::operator[](int x){
    if(x>(this->nodeCount()))
        throw invalid_argument( "Subscript out of range" );
    else{
        node<T>* temp;

        temp=ptr;

        while(x>0){
            temp=temp->link;
            --x;

        }
        return temp->data;
    }
}


template <class T>

class Qeue{
public:
    Qeue(int =100);
    ~Qeue();

    void Add(T);
    T Del(void);

    int isfull(void);
    int isempty(void);

private:
    int	maxsize;
    int front;
    int rare;
    LinkedList<T> list;
};

template <class T>
Qeue<T>::Qeue(int x){
    front=0;
    rare=0;
    maxsize=x;
}

template <class T>
Qeue<T>::~Qeue(){
}

template <class T>
void Qeue<T>::Add(T data){
    ++rare;
    rare=(rare%maxsize);
    list.Add(data);
}

template <class T>
T Qeue<T>::Del(){
    T data=0;
    data=list.ptr->data;
    list.Del(list.ptr->data);
    return data;
}

template <typename T>
int Qeue<T>::isempty(){
    if(rare==front)
        return 1;
    return 0;
}


template <typename T>

class Stack
{
private:
    int top;
    int len;
    LinkedList<T> list;

public:
    Stack(int n){
        top=0;
        len=n;
    }
    ~Stack(){

    }

    void push(T);
    T pop();

    int isempty(void);
    int isfull(void);
};


template <typename T>
void Stack<T>::push(T data){
    list.Add(data);
    ++top;
}

template <typename T>
T Stack<T>::pop(){
    T data;
    --top;
    data=list[top];
    list.Del(list[top]);
    return data;
}

template <typename T>
int Stack<T>::isempty(void){
    if(top==0)
        return 1;
    return 0;
}

template <typename T>
int Stack<T>::isfull(void){
    if(top==len)
        return 1;
    return 0;
}