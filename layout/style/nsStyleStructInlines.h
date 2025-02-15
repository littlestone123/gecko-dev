/* vim: set shiftwidth=2 tabstop=8 autoindent cindent expandtab: */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*
 * Inline methods that belong in nsStyleStruct.h, except that they
 * require more headers.
 */

#ifndef nsStyleStructInlines_h_
#define nsStyleStructInlines_h_

#include "nsIFrame.h"
#include "nsStyleStruct.h"
#include "nsIContent.h" // for GetParent()
#include "nsTextFrame.h" // for nsTextFrame::ShouldSuppressLineBreak

inline void
nsStyleImage::EnsureCachedBIData() const
{
  if (!mCachedBIData) {
    const_cast<nsStyleImage*>(this)->mCachedBIData =
      mozilla::MakeUnique<CachedBorderImageData>();
  }
}

inline void
nsStyleImage::SetSubImage(uint8_t aIndex, imgIContainer* aSubImage) const
{
  EnsureCachedBIData();
  mCachedBIData->SetSubImage(aIndex, aSubImage);
}

inline imgIContainer*
nsStyleImage::GetSubImage(uint8_t aIndex) const
{
  return (mCachedBIData) ? mCachedBIData->GetSubImage(aIndex) : nullptr;
}

bool
nsStyleText::HasTextShadow() const
{
  return mTextShadow;
}

nsCSSShadowArray*
nsStyleText::GetTextShadow() const
{
  return mTextShadow;
}

bool
nsStyleText::NewlineIsSignificant(const nsTextFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleText() == this, "unexpected aContextFrame");
  return NewlineIsSignificantStyle() &&
         !aContextFrame->ShouldSuppressLineBreak() &&
         !aContextFrame->StyleContext()->IsTextCombined();
}

bool
nsStyleText::WhiteSpaceCanWrap(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleText() == this, "unexpected aContextFrame");
  return WhiteSpaceCanWrapStyle() && !aContextFrame->IsSVGText() &&
         !aContextFrame->StyleContext()->IsTextCombined();
}

bool
nsStyleText::WordCanWrap(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleText() == this, "unexpected aContextFrame");
  return WordCanWrapStyle() && !aContextFrame->IsSVGText();
}

bool
nsStyleDisplay::IsBlockInside(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  if (aContextFrame->IsSVGText()) {
    return aContextFrame->GetType() == nsGkAtoms::blockFrame;
  }
  return IsBlockInsideStyle();
}

bool
nsStyleDisplay::IsBlockOutside(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  if (aContextFrame->IsSVGText()) {
    return aContextFrame->GetType() == nsGkAtoms::blockFrame;
  }
  return IsBlockOutsideStyle();
}

bool
nsStyleDisplay::IsInlineOutside(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  if (aContextFrame->IsSVGText()) {
    return aContextFrame->GetType() != nsGkAtoms::blockFrame;
  }
  return IsInlineOutsideStyle();
}

bool
nsStyleDisplay::IsOriginalDisplayInlineOutside(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  if (aContextFrame->IsSVGText()) {
    return aContextFrame->GetType() != nsGkAtoms::blockFrame;
  }
  return IsOriginalDisplayInlineOutsideStyle();
}

uint8_t
nsStyleDisplay::GetDisplay(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  if (aContextFrame->IsSVGText() &&
      mDisplay != NS_STYLE_DISPLAY_NONE) {
    return aContextFrame->GetType() == nsGkAtoms::blockFrame ?
             NS_STYLE_DISPLAY_BLOCK :
             NS_STYLE_DISPLAY_INLINE;
  }
  return mDisplay;
}

bool
nsStyleDisplay::IsFloating(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  return IsFloatingStyle() && !aContextFrame->IsSVGText();
}

// If you change this function, also change the corresponding block in
// nsCSSFrameConstructor::ConstructFrameFromItemInternal that references
// this function in comments.
bool
nsStyleDisplay::HasTransform(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  return HasTransformStyle() && aContextFrame->IsFrameOfType(nsIFrame::eSupportsCSSTransforms);
}

template<class StyleContextLike>
bool
nsStyleDisplay::HasFixedPosContainingBlockStyleInternal(
                  StyleContextLike* aStyleContext) const
{
  // NOTE: Any CSS properties that influence the output of this function
  // should have the CSS_PROPERTY_FIXPOS_CB set on them.
  NS_ASSERTION(aStyleContext->StyleDisplay() == this,
               "unexpected aStyleContext");
  return IsContainPaint() ||
         HasPerspectiveStyle() ||
         (mWillChangeBitField & NS_STYLE_WILL_CHANGE_FIXPOS_CB) ||
         aStyleContext->StyleEffects()->HasFilters();
}

template<class StyleContextLike>
bool
nsStyleDisplay::IsFixedPosContainingBlockForAppropriateFrame(
                  StyleContextLike* aStyleContext) const
{
  // NOTE: Any CSS properties that influence the output of this function
  // should have the CSS_PROPERTY_FIXPOS_CB set on them.
  return HasFixedPosContainingBlockStyleInternal(aStyleContext) ||
         HasTransformStyle();
}

bool
nsStyleDisplay::IsFixedPosContainingBlock(const nsIFrame* aContextFrame) const
{
  // NOTE: Any CSS properties that influence the output of this function
  // should have the CSS_PROPERTY_FIXPOS_CB set on them.
  if (!HasFixedPosContainingBlockStyleInternal(aContextFrame->StyleContext()) &&
      !HasTransform(aContextFrame)) {
    return false;
  }
  return !aContextFrame->IsSVGText();
}

template<class StyleContextLike>
bool
nsStyleDisplay::HasAbsPosContainingBlockStyleInternal(
                  StyleContextLike* aStyleContext) const
{
  // NOTE: Any CSS properties that influence the output of this function
  // should have the CSS_PROPERTY_ABSPOS_CB set on them.
  NS_ASSERTION(aStyleContext->StyleDisplay() == this,
               "unexpected aStyleContext");
  return IsAbsolutelyPositionedStyle() ||
         IsRelativelyPositionedStyle() ||
         (mWillChangeBitField & NS_STYLE_WILL_CHANGE_ABSPOS_CB);
}

template<class StyleContextLike>
bool
nsStyleDisplay::IsAbsPosContainingBlockForAppropriateFrame(StyleContextLike* aStyleContext) const
{
  // NOTE: Any CSS properties that influence the output of this function
  // should have the CSS_PROPERTY_ABSPOS_CB set on them.
  return HasAbsPosContainingBlockStyleInternal(aStyleContext) ||
         IsFixedPosContainingBlockForAppropriateFrame(aStyleContext);
}

bool
nsStyleDisplay::IsAbsPosContainingBlock(const nsIFrame* aContextFrame) const
{
  // NOTE: Any CSS properties that influence the output of this function
  // should have the CSS_PROPERTY_ABSPOS_CB set on them.
  nsStyleContext* sc = aContextFrame->StyleContext();
  if (!HasAbsPosContainingBlockStyleInternal(sc) &&
      !HasFixedPosContainingBlockStyleInternal(sc) &&
      !HasTransform(aContextFrame)) {
    return false;
  }
  return !aContextFrame->IsSVGText();
}

bool
nsStyleDisplay::IsRelativelyPositioned(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  return IsRelativelyPositionedStyle() && !aContextFrame->IsSVGText();
}

bool
nsStyleDisplay::IsAbsolutelyPositioned(const nsIFrame* aContextFrame) const
{
  NS_ASSERTION(aContextFrame->StyleDisplay() == this, "unexpected aContextFrame");
  return IsAbsolutelyPositionedStyle() && !aContextFrame->IsSVGText();
}

uint8_t
nsStyleUserInterface::GetEffectivePointerEvents(nsIFrame* aFrame) const
{
  if (aFrame->GetContent() && !aFrame->GetContent()->GetParent()) {
    // The root element has a cluster of frames associated with it
    // (root scroll frame, canvas frame, the actual primary frame). Make
    // those take their pointer-events value from the root element's primary
    // frame.
    nsIFrame* f = aFrame->GetContent()->GetPrimaryFrame();
    if (f) {
      return f->StyleUserInterface()->mPointerEvents;
    }
  }
  return mPointerEvents;
}

bool
nsStyleBackground::HasLocalBackground() const
{
  NS_FOR_VISIBLE_IMAGE_LAYERS_BACK_TO_FRONT(i, mImage) {
    const nsStyleImageLayers::Layer& layer = mImage.mLayers[i];
    if (!layer.mImage.IsEmpty() &&
        layer.mAttachment == NS_STYLE_IMAGELAYER_ATTACHMENT_LOCAL) {
      return true;
    }
  }
  return false;
}

#endif /* !defined(nsStyleStructInlines_h_) */
