# Summary of the Stabilizer Operator Pool Idea

This summary incorporates the correction to our earlier reasoning: **the original invariance condition establishes that \(\hat T_i^\dagger\) preserves \(V_k\), while the stabilizer-pool definition establishes that \(\hat T_i\) preserves \(V_k\). Together, these conditions guarantee that the mapped unitary preserves both \(V_k\) and \(V_k^\perp\).**

## 1. Reference states and subspaces

Consider \(k+1\) orthonormal reference states,

\[
\{|\phi_0\rangle,|\phi_1\rangle,\ldots,|\phi_k\rangle\}.
\]

For the selected reference state \(|\phi_k\rangle\), define

\[
V_k=\operatorname{span}\{|\phi_k\rangle\},
\]

and its orthogonal complement,

\[
V_k^\perp=
\left\{
|\psi\rangle:
\langle\phi_k|\psi\rangle=0
\right\}.
\]

Orthonormality ensures that

\[
\operatorname{span}
\{|\phi_0\rangle,\ldots,|\phi_{k-1}\rangle\}
\subseteq V_k^\perp.
\]

The objective is to construct an operator pool whose unitary images preserve the selected reference subspace and its orthogonal complement.

## 2. Original operator pool and its invariance condition

Let the original operator pool be

\[
\mathcal O=\{\hat T_i\}.
\]

Assume that every operator in this pool preserves the orthogonal complement:

\[
\boxed{
\hat T_iV_k^\perp\subseteq V_k^\perp,
\qquad \forall\,\hat T_i\in\mathcal O.
}
\]

Equivalently,

\[
\langle\phi_k|\hat T_i|\psi\rangle=0
\qquad
\forall\,|\psi\rangle\in V_k^\perp.
\]

Taking the adjoint relation gives

\[
\boxed{
\hat T_i^\dagger|\phi_k\rangle\in V_k.
}
\]

Since \(V_k\) is one-dimensional, there exists a scalar \(\mu_i\) such that

\[
\hat T_i^\dagger|\phi_k\rangle
=\mu_i|\phi_k\rangle.
\]

**Important:** this original assumption does not, by itself, imply that \(\hat T_i|\phi_k\rangle\in V_k\).

## 3. Definition of the stabilizer operator pool

The proposed construction selects those operators from the original pool that preserve the reference subspace:

\[
\boxed{
\mathcal O_k=
\left\{
\hat T_i\in\mathcal O:
\hat T_i|\phi_k\rangle\in V_k
\right\}.
}
\]

Equivalently,

\[
\boxed{
\hat T_i|\phi_k\rangle
=\lambda_i|\phi_k\rangle,
\qquad \hat T_i\in\mathcal O_k.
}
\]

This is the **stabilizer operator pool** associated with the reference state \(|\phi_k\rangle\).

By definition, every selected operator preserves \(V_k\). Combined with the original pool's invariance condition, we have

\[
\boxed{
\begin{aligned}
\hat T_i|\phi_k\rangle
&=\lambda_i|\phi_k\rangle,\\
\hat T_i^\dagger|\phi_k\rangle
&=\mu_i|\phi_k\rangle.
\end{aligned}
}
\]

Both the operator and its adjoint therefore preserve the one-dimensional reference subspace.

## 4. Unitary mapping

Map each operator in the stabilizer operator pool to a unitary using

\[
\boxed{
\hat U_i(\theta)
=e^{\theta(\hat T_i-\hat T_i^\dagger)},
\qquad \theta\in\mathbb R.
}
\]

Define the generator

\[
\hat A_i=\hat T_i-\hat T_i^\dagger.
\]

It is anti-Hermitian:

\[
\hat A_i^\dagger=-\hat A_i.
\]

Consequently,

\[
\hat U_i(\theta)^\dagger\hat U_i(\theta)=I.
\]

Thus, \(\hat U_i(\theta)\) is unitary for real \(\theta\).

## 5. Why the mapped unitary stabilizes the reference state

For \(\hat T_i\in\mathcal O_k\), both \(\hat T_i\) and \(\hat T_i^\dagger\) have \(|\phi_k\rangle\) as an eigenvector. Hence,

\[
\begin{aligned}
\hat A_i|\phi_k\rangle
&=(\hat T_i-\hat T_i^\dagger)|\phi_k\rangle\\
&=(\lambda_i-\mu_i)|\phi_k\rangle.
\end{aligned}
\]

Because \(\hat A_i\) is anti-Hermitian, its eigenvalue \(\lambda_i-\mu_i\) is purely imaginary. Write it as \(i\gamma_i\), where \(\gamma_i\in\mathbb R\).

Therefore,

\[
\begin{aligned}
\hat U_i(\theta)|\phi_k\rangle
&=e^{\theta\hat A_i}|\phi_k\rangle\\
&=e^{i\theta\gamma_i}|\phi_k\rangle.
\end{aligned}
\]

The unitary maps the reference state to itself up to a global phase.

Define the state stabilizer subgroup as

\[
S_k=
\left\{
U\in\mathcal U(\mathcal H):
U|\phi_k\rangle=e^{i\alpha}|\phi_k\rangle,
\quad \alpha\in\mathbb R
\right\}.
\]

We obtain the desired inclusion:

\[
\boxed{
\mathcal U_\theta(\mathcal O_k)\subseteq S_k.
}
\]

Here, \(\mathcal U_\theta(\mathcal O_k)\) denotes the set of unitary images of the operators in \(\mathcal O_k\), for a fixed real \(\theta\).

## 6. Why the orthogonal complement is also preserved

The previous conclusion establishes that the mapped unitary preserves \(V_k\). We can also establish preservation of \(V_k^\perp\).

Since \(\hat T_i\in\mathcal O_k\),

\[
\hat T_i|\phi_k\rangle\in V_k.
\]

This implies

\[
\hat T_i^\dagger V_k^\perp\subseteq V_k^\perp.
\]

Together with the original assumption,

\[
\hat T_iV_k^\perp\subseteq V_k^\perp,
\]

we conclude that

\[
\boxed{
(\hat T_i-\hat T_i^\dagger)V_k^\perp
\subseteq V_k^\perp.
}
\]

Therefore, its exponential preserves the orthogonal complement:

\[
\hat U_i(\theta)V_k^\perp
\subseteq V_k^\perp.
\]

Because \(\hat U_i(\theta)\) is unitary and the Hilbert space is finite-dimensional, this inclusion is an equality:

\[
\boxed{
\hat U_i(\theta)V_k^\perp=V_k^\perp.
}
\]

Hence, the mapped unitary preserves both subspaces:

\[
\boxed{
\begin{aligned}
\hat U_i(\theta)V_k&=V_k,\\
\hat U_i(\theta)V_k^\perp&=V_k^\perp.
\end{aligned}
}
\]

This is the corrected conclusion under the stated assumptions.

## 7. Complete construction at a glance

1. **Original operator pool**
   \[
   \mathcal O=\{\hat T_i\},\qquad
   \hat T_iV_k^\perp\subseteq V_k^\perp.
   \]

2. **Select stabilizer operators**
   \[
   \mathcal O_k=
   \{\hat T_i\in\mathcal O:\hat T_i|\phi_k\rangle\in V_k\}.
   \]

3. **Apply the unitary map**
   \[
   \hat U_i(\theta)=e^{\theta(\hat T_i-\hat T_i^\dagger)}.
   \]

4. **Result**
   \[
   \mathcal U_\theta(\mathcal O_k)\subseteq S_k,
   \]
   \[
   \hat U_iV_k=V_k,\qquad
   \hat U_iV_k^\perp=V_k^\perp.
   \]

## 8. Final statement of the idea

The proposed method constructs a stabilizer operator pool by selecting operators from an original pool that preserve the one-dimensional subspace spanned by a chosen reference state. The original pool is assumed to preserve the orthogonal complement of that state. These two conditions together ensure that both the operators and their adjoints preserve the reference subspace. Consequently, the anti-Hermitian generators \(\hat T_i-\hat T_i^\dagger\) preserve both subspaces, and their unitary exponentials stabilize the reference state up to a global phase while preserving its orthogonal complement.

**The central result is**

\[
\boxed{
\mathcal U_\theta(\mathcal O_k)
\subseteq S_k,
\qquad
\hat U_iV_k^\perp=V_k^\perp.
}
\]

This conclusion is established under the stated assumptions. It does not yet establish that the stabilizer operator pool is nonempty, that it is maximal, or that the construction is computationally advantageous; those are separate questions for further investigation.
