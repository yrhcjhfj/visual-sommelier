"""Explanation service for generating device explanations and instructions."""

import logging
from typing import List, Optional
import re

from ..models.explanation import Explanation, Step
from ..models.device import DeviceContext, BoundingBox
from ..adapters.factory import AdapterFactory

logger = logging.getLogger(__name__)


class ExplanationService:
    """Service for generating explanations and instructions using LLM."""
    
    # Safety-related keywords that trigger warnings
    ELECTRICAL_KEYWORDS = [
        "electric", "electrical", "power", "voltage", "current",
        "plug", "socket", "outlet", "wire", "cable", "battery"
    ]
    
    DANGEROUS_OPERATION_KEYWORDS = [
        "repair", "disassemble", "open", "remove", "disconnect",
        "high temperature", "hot", "heat", "sharp", "cutting"
    ]
    
    # Device types that require electrical warnings
    ELECTRICAL_DEVICES = [
        "washing_machine", "microwave", "oven", "dishwasher",
        "air_conditioner", "tv", "coffee_machine", "vacuum_cleaner",
        "refrigerator", "toaster"
    ]
    
    # Disclaimer text
    DISCLAIMER = (
        "⚠️ Disclaimer: This information is provided for educational purposes only "
        "and does not replace the official manufacturer's instructions. "
        "Always refer to the device's manual for complete and accurate information."
    )

    def __init__(self):
        """Initialize the explanation service."""
        self.adapter_factory = AdapterFactory()
        self._llm_adapter = None
    
    def _get_llm_adapter(self):
        """Get or create LLM adapter."""
        if self._llm_adapter is None:
            self._llm_adapter = self.adapter_factory.get_llm_adapter("llava")
        return self._llm_adapter
    
    def _detect_safety_concerns(
        self,
        device_context: DeviceContext,
        question: str = ""
    ) -> List[str]:
        """Detect potential safety concerns based on device and question.
        
        Args:
            device_context: Context about the device
            question: User's question or task description
            
        Returns:
            List of safety warnings
        """
        warnings = []
        
        # Check if device is electrical
        if device_context.device_type in self.ELECTRICAL_DEVICES:
            warnings.append(
                "⚠️ ELECTRICAL DEVICE: Always unplug the device before performing "
                "any maintenance or cleaning. Risk of electric shock."
            )
        
        # Check for dangerous operations in question
        question_lower = question.lower()
        for keyword in self.DANGEROUS_OPERATION_KEYWORDS:
            if keyword in question_lower:
                warnings.append(
                    "⚠️ CAUTION: This operation may involve risks. "
                    "If you are unsure, consult a professional technician."
                )
                break
        
        # Add device-specific warnings from context
        if device_context.safety_warnings:
            warnings.extend(device_context.safety_warnings)
        
        return warnings
    
    def _add_disclaimer(self, text: str) -> str:
        """Add disclaimer to explanation text.
        
        Args:
            text: Original explanation text
            
        Returns:
            Text with disclaimer appended
        """
        return f"{text}\n\n{self.DISCLAIMER}"
    
    def generate_explanation(
        self,
        image: bytes,
        question: str,
        device_context: DeviceContext,
        language: str = "en"
    ) -> Explanation:
        """Generate explanation for a device-related question.
        
        Implements Requirements 2.1, 2.2, 2.3, 8.1, 8.2, 8.5.
        
        Args:
            image: Device image as bytes
            question: User's question about the device
            device_context: Context information about the device
            language: Language code (en, ru, zh)
            
        Returns:
            Explanation object with text, warnings, and metadata
        """
        logger.info(f"Generating explanation for question: {question[:50]}...")
        
        try:
            # Detect safety concerns
            warnings = self._detect_safety_concerns(device_context, question)
            
            # Build prompt for LLM
            prompt = self._build_explanation_prompt(
                question=question,
                device_context=device_context,
                warnings=warnings
            )
            
            # Get LLM adapter and generate response
            llm = self._get_llm_adapter()
            response_text = llm.generate_completion(
                prompt=prompt,
                image=image,
                language=language,
                max_tokens=512
            )
            
            # Add disclaimer
            response_text = self._add_disclaimer(response_text)
            
            # Create explanation object
            explanation = Explanation(
                text=response_text,
                steps=None,  # No steps for simple explanations
                warnings=warnings,
                confidence=0.8,  # Default confidence
                sources=["LLaVA Vision-Language Model"]
            )
            
            logger.info("Explanation generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Return a fallback explanation
            return Explanation(
                text=self._add_disclaimer(
                    "I apologize, but I encountered an error while analyzing the device. "
                    "Please try again or rephrase your question."
                ),
                warnings=["⚠️ Error occurred during analysis"],
                confidence=0.0,
                sources=[]
            )
    
    def _build_explanation_prompt(
        self,
        question: str,
        device_context: DeviceContext,
        warnings: List[str]
    ) -> str:
        """Build prompt for explanation generation.
        
        Args:
            question: User's question
            device_context: Device context
            warnings: Safety warnings
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"You are an expert assistant helping users understand household devices.",
            f"",
            f"Device Type: {device_context.device_type}",
        ]
        
        if device_context.brand:
            prompt_parts.append(f"Brand: {device_context.brand}")
        if device_context.model:
            prompt_parts.append(f"Model: {device_context.model}")
        
        if device_context.detected_controls:
            prompt_parts.append(f"Detected Controls: {len(device_context.detected_controls)} elements")
        
        prompt_parts.extend([
            f"",
            f"User Question: {question}",
            f"",
            f"Please provide a clear, helpful explanation that:",
            f"1. Directly answers the user's question",
            f"2. Uses simple, easy-to-understand language",
            f"3. Includes practical tips if relevant",
            f"4. Is concise but complete",
        ])
        
        if warnings:
            prompt_parts.append(f"5. Acknowledges any safety concerns mentioned")
        
        return "\n".join(prompt_parts)

    def generate_instructions(
        self,
        task: str,
        device_context: DeviceContext,
        language: str = "en",
        image: Optional[bytes] = None
    ) -> List[Step]:
        """Generate step-by-step instructions for a task.
        
        Implements Requirements 3.1, 3.5, 8.1, 8.2, 8.5.
        
        Args:
            task: Description of the task to perform
            device_context: Context information about the device
            language: Language code (en, ru, zh)
            image: Optional device image for visual context
            
        Returns:
            List of Step objects with instructions
        """
        logger.info(f"Generating instructions for task: {task[:50]}...")
        
        try:
            # Detect safety concerns
            warnings = self._detect_safety_concerns(device_context, task)
            
            # Build prompt for instruction generation
            prompt = self._build_instructions_prompt(
                task=task,
                device_context=device_context,
                warnings=warnings
            )
            
            # Get LLM adapter and generate response
            llm = self._get_llm_adapter()
            response_text = llm.generate_completion(
                prompt=prompt,
                image=image,
                language=language,
                max_tokens=1024  # More tokens for detailed instructions
            )
            
            # Parse response into steps
            steps = self._parse_steps_from_text(response_text, warnings)
            
            logger.info(f"Generated {len(steps)} instruction steps")
            return steps
            
        except Exception as e:
            logger.error(f"Error generating instructions: {e}")
            # Return a fallback step
            return [
                Step(
                    number=1,
                    description=(
                        "I apologize, but I encountered an error while generating instructions. "
                        "Please try again or consult the device manual."
                    ),
                    warning="⚠️ Error occurred during instruction generation"
                )
            ]
    
    def _build_instructions_prompt(
        self,
        task: str,
        device_context: DeviceContext,
        warnings: List[str]
    ) -> str:
        """Build prompt for instruction generation.
        
        Args:
            task: Task description
            device_context: Device context
            warnings: Safety warnings
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"You are an expert assistant helping users with household devices.",
            f"",
            f"Device Type: {device_context.device_type}",
        ]
        
        if device_context.brand:
            prompt_parts.append(f"Brand: {device_context.brand}")
        if device_context.model:
            prompt_parts.append(f"Model: {device_context.model}")
        
        prompt_parts.extend([
            f"",
            f"Task: {task}",
            f"",
            f"Please provide step-by-step instructions in the following format:",
            f"",
            f"Step 1: [First action to take]",
            f"Step 2: [Second action to take]",
            f"Step 3: [Third action to take]",
            f"...",
            f"",
            f"Requirements:",
            f"- Use clear, numbered steps",
            f"- Each step should be a single, specific action",
            f"- Use simple language that anyone can understand",
            f"- Include safety warnings where appropriate",
            f"- Be practical and actionable",
        ])
        
        if warnings:
            prompt_parts.extend([
                f"",
                f"IMPORTANT SAFETY NOTES:",
            ])
            for warning in warnings:
                prompt_parts.append(f"- {warning}")
        
        return "\n".join(prompt_parts)
    
    def _parse_steps_from_text(
        self,
        text: str,
        safety_warnings: List[str]
    ) -> List[Step]:
        """Parse step-by-step instructions from LLM response text.
        
        Args:
            text: Response text from LLM
            safety_warnings: General safety warnings to check against
            
        Returns:
            List of Step objects
        """
        steps = []
        
        # Pattern to match numbered steps (e.g., "Step 1:", "1.", "1)")
        step_pattern = re.compile(
            r'(?:Step\s+)?(\d+)[\.:)]\s*(.+?)(?=(?:Step\s+)?\d+[\.:)]|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        matches = step_pattern.findall(text)
        
        if not matches:
            # Fallback: split by newlines and look for numbered items
            lines = text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit():
                    # Extract number and description
                    parts = re.split(r'[\.:)]\s*', line, maxsplit=1)
                    if len(parts) == 2:
                        try:
                            number = int(parts[0])
                            description = parts[1].strip()
                            if description:
                                matches.append((str(number), description))
                        except ValueError:
                            continue
        
        # Create Step objects
        for i, (num_str, description) in enumerate(matches, start=1):
            try:
                step_number = int(num_str)
            except ValueError:
                step_number = i
            
            # Clean up description
            description = description.strip()
            
            # Check if this step has safety concerns
            step_warning = None
            desc_lower = description.lower()
            
            # Check for dangerous keywords in this specific step
            for keyword in self.DANGEROUS_OPERATION_KEYWORDS:
                if keyword in desc_lower:
                    step_warning = "⚠️ CAUTION: This step requires care. Follow instructions carefully."
                    break
            
            # Check for electrical keywords
            if not step_warning:
                for keyword in self.ELECTRICAL_KEYWORDS:
                    if keyword in desc_lower:
                        step_warning = "⚠️ ELECTRICAL: Ensure device is unplugged before this step."
                        break
            
            step = Step(
                number=step_number,
                description=description,
                warning=step_warning,
                highlighted_area=None,  # Could be enhanced with CV in future
                completed=False
            )
            steps.append(step)
        
        # If no steps were parsed, create a single step with the full text
        if not steps:
            steps.append(
                Step(
                    number=1,
                    description=text.strip(),
                    warning=safety_warnings[0] if safety_warnings else None
                )
            )
        
        return steps
    
    def clarify_step(
        self,
        step: Step,
        question: str,
        device_context: DeviceContext,
        language: str = "en",
        image: Optional[bytes] = None
    ) -> str:
        """Provide additional clarification for a specific step.
        
        Implements Requirement 3.5.
        
        Args:
            step: The step that needs clarification
            question: User's specific question about the step
            device_context: Context information about the device
            language: Language code (en, ru, zh)
            image: Optional device image for visual context
            
        Returns:
            Clarification text
        """
        logger.info(f"Generating clarification for step {step.number}")
        
        try:
            # Build prompt for clarification
            prompt = self._build_clarification_prompt(
                step=step,
                question=question,
                device_context=device_context
            )
            
            # Get LLM adapter and generate response
            llm = self._get_llm_adapter()
            response_text = llm.generate_completion(
                prompt=prompt,
                image=image,
                language=language,
                max_tokens=512
            )
            
            # Add disclaimer
            response_text = self._add_disclaimer(response_text)
            
            logger.info("Clarification generated successfully")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating clarification: {e}")
            return self._add_disclaimer(
                "I apologize, but I encountered an error while generating clarification. "
                "Please try rephrasing your question or consult the device manual."
            )
    
    def _build_clarification_prompt(
        self,
        step: Step,
        question: str,
        device_context: DeviceContext
    ) -> str:
        """Build prompt for step clarification.
        
        Args:
            step: Step to clarify
            question: User's question
            device_context: Device context
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"You are an expert assistant helping users understand device instructions.",
            f"",
            f"Device Type: {device_context.device_type}",
        ]
        
        if device_context.brand:
            prompt_parts.append(f"Brand: {device_context.brand}")
        
        prompt_parts.extend([
            f"",
            f"The user is working on Step {step.number}:",
            f'"{step.description}"',
            f"",
            f"User's Question: {question}",
            f"",
            f"Please provide a clear, detailed clarification that:",
            f"1. Directly addresses the user's question",
            f"2. Provides additional context or details about this step",
            f"3. Uses simple, easy-to-understand language",
            f"4. Includes practical tips if helpful",
            f"5. Keeps the response focused on this specific step",
        ])
        
        if step.warning:
            prompt_parts.extend([
                f"",
                f"SAFETY NOTE: {step.warning}",
                f"Make sure to acknowledge any safety concerns in your response.",
            ])
        
        return "\n".join(prompt_parts)
